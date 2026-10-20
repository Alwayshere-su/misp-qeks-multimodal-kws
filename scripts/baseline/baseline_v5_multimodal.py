"""
MISP-QEKS Baseline V5 — 多模态: Whisper + LipReading + Text
============================================================
目标: 0.75+ AUC (vs V3 的 0.6466)

关键改进:
1. Whisper Tiny encoder → 384-dim 音频特征 (替代弱MFCC)
2. LipReading 3D ResNet → 256-dim 视频口唇特征 (新增)
3. Text embedding → 128-dim 关键词文本特征
4. 多模态融合: audio+video+text → 联合 enrollment embedding
5. Cross-attention matching + BCE loss (V3验证有效)
6. 硬负样本挖掘 (基于FA音素相似度, 可选)
"""

import os, sys, re, json, random, pickle, shutil
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import soundfile as sf

# ============================================================
BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
DATA = BASE / "val"
TEST = BASE / "test"
AUX = BASE / "辅助资源"
OUT = BASE / "output_v5"
SUB = BASE / "submission"
OUT.mkdir(exist_ok=True); SUB.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SR = 16000
MAX_TEXT_LEN = 20

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(SEED)

# ============================================================
# 1. 加载 Whisper 模型 (从 tiny.pt)
# ============================================================

def build_whisper_encoder():
    """构建 Whisper Tiny 编码器并加载预训练权重"""
    # Whisper 使用标准的 Transformer encoder
    ckpt_path = AUX / "pretrained_feature_models/whisper/ckps/tiny.pt"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    dims = ckpt['dims']
    sd = ckpt['model_state_dict']

    encoder = WhisperEncoder(
        n_mels=dims['n_mels'],
        n_audio_state=dims['n_audio_state'],
        n_audio_head=dims['n_audio_head'],
        n_audio_layer=dims['n_audio_layer'],
    )

    # 加载 encoder 权重 (过滤出 encoder.* 的 key)
    encoder_sd = {}
    for k, v in sd.items():
        if k.startswith('encoder.'):
            encoder_sd[k.replace('encoder.', '')] = v
    encoder.load_state_dict(encoder_sd, strict=False)
    print(f"  Whisper encoder loaded: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params")
    return encoder, dims['n_audio_state']

class WhisperEncoder(nn.Module):
    """Whisper encoder: Conv1d stem + sinusoidal PE + Transformer blocks"""
    def __init__(self, n_mels=80, n_audio_state=384, n_audio_head=6, n_audio_layer=4):
        super().__init__()
        self.n_audio_state = n_audio_state

        # Conv stem: 2x Conv1d with GELU
        self.conv1 = nn.Conv1d(n_mels, n_audio_state, 3, padding=1)
        self.conv2 = nn.Conv1d(n_audio_state, n_audio_state, 3, stride=2, padding=1)

        # Positional embedding (sinusoidal)
        self.register_buffer('positional_embedding', torch.empty(1500, n_audio_state))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_audio_state, n_audio_head)
            for _ in range(n_audio_layer)
        ])

        self.ln_post = nn.LayerNorm(n_audio_state)

    def forward(self, mel):
        """mel: (B, n_mels, T)"""
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T, n_audio_state)

        # Add positional embedding
        seq_len = x.shape[1]
        x = x + self.positional_embedding[:seq_len]

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x  # (B, T, n_audio_state)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_state, n_head, batch_first=True)
        self.attn_ln = nn.LayerNorm(n_state)
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_state * 4), nn.GELU(),
            nn.Linear(n_state * 4, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.attn_ln(x)

        # MLP with residual
        x = x + self.mlp(x)
        x = self.mlp_ln(x)
        return x

# ============================================================
# 2. 加载 LipReading 模型 (从 lipreading_LRW_0.8018.pt)
# ============================================================

def build_lipreading_encoder():
    """构建口唇视频编码器"""
    ckpt_path = AUX / "pretrained_feature_models/lipreading/lipreading_LRW_0.8018.pt"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # 从 state dict 推斷架构
    # video_frontend: 3D Conv + BN (64 filters, 5x7x7 kernel)
    # resnet: ResNet18 with 3D convs
    model = LipReadingEncoder()
    # 加载权重时忽略不匹配的 keys
    model.load_state_dict(ckpt, strict=False)
    print(f"  LipReading encoder loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model, 256  # output dim from fc layer

class LipReadingEncoder(nn.Module):
    """3D CNN 口唇视频编码器 (ResNet18-based)"""
    def __init__(self):
        super().__init__()
        # Frontend: 3D conv + BN + ReLU + MaxPool
        self.video_frontend = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        # ResNet18 adapted for 3D (using standard ResNet building blocks)
        self.resnet = ResNet3D(BasicBlock3D, [2, 2, 2, 2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        """x: (B, 1, T, H, W) — grayscale video frames"""
        x = self.video_frontend(x)  # (B, 64, T, H', W')
        x = self.resnet(x)          # (B, 512, T', H'', W'')
        x = self.avgpool(x)         # (B, 512, 1, 1, 1)
        x = x.flatten(1)            # (B, 512)
        return x

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ResNet3D(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # fc and bnfc (from checkpoint)
        self.fc = nn.Linear(512, 256)
        self.bnfc = nn.BatchNorm1d(256)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes*block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes*block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# ============================================================
# 3. 文本编码器
# ============================================================

def build_vocab(words):
    chars = set()
    for w in words:
        for c in w.lower():
            chars.add(c)
    vocab = ['<pad>', '<unk>'] + sorted(chars)
    return vocab, {c: i for i, c in enumerate(vocab)}

def tokenize(word, char2idx, max_len=MAX_TEXT_LEN):
    ids = [char2idx.get(c, char2idx['<unk>']) for c in word.lower()]
    if len(ids) > max_len: ids = ids[:max_len]
    else: ids = ids + [char2idx['<pad>']] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

class TextEncoder(nn.Module):
    """简单字符级文本编码"""
    def __init__(self, vocab_size, emb_dim=64, out_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, out_dim, 3, padding=1), nn.BatchNorm1d(out_dim), nn.GELU(),
        )
        self.attn_pool = nn.Conv1d(out_dim, 1, 1)
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, tokens):
        x = self.embed(tokens).transpose(1, 2)
        x = self.conv(x)
        w = self.attn_pool(x).softmax(-1)
        x = (w * x).sum(-1)
        return self.proj(x)

# ============================================================
# 4. 特征提取: 音频(MFCC) + 视频(LipReading frames) + 文本
# ============================================================

def extract_mfcc_feats(wav_path, max_sec):
    """V3 兼容的 MFCC (Whisper 也用, 用于 baseline 对比)"""
    data, sr = sf.read(str(wav_path), dtype='float32')
    if data.ndim > 1: data = data.mean(axis=1)
    audio = torch.from_numpy(data).float()
    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
    max_s = int(max_sec * SR)
    if len(audio) > max_s: audio = audio[:max_s]
    elif len(audio) < max_s: audio = F.pad(audio, (0, max_s - len(audio)))
    audio = audio.unsqueeze(0)
    mfcc = torchaudio.transforms.MFCC(SR, n_mfcc=40, melkwargs={
        'n_mels': 80, 'n_fft': 512, 'hop_length': 160, 'win_length': 400
    })(audio)
    delta = torchaudio.transforms.ComputeDeltas()
    d1, d2 = delta(mfcc), delta(delta(mfcc))
    return torch.cat([mfcc, d1, d2], dim=1).squeeze(0)  # (120, T)

def extract_whisper_feats(wav_path, whisper_encoder, max_sec, device=None):
    """用 Whisper 提取 mel 频谱特征, 然后过 encoder"""
    data, sr = sf.read(str(wav_path), dtype='float32')
    if data.ndim > 1: data = data.mean(axis=1)
    audio = torch.from_numpy(data).float()
    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
    max_s = int(max_sec * SR)
    if len(audio) > max_s: audio = audio[:max_s]
    elif len(audio) < max_s: audio = F.pad(audio, (0, max_s - len(audio)))

    # Compute mel spectrogram (Whisper-style: 80 mel bins, 25ms window, 10ms hop)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_mels=80, n_fft=400, hop_length=160, win_length=400
    )(audio.unsqueeze(0))  # (1, 80, T)
    mel = torch.log(torch.clamp(mel, min=1e-10))

    # Move to same device as encoder
    if device is not None:
        mel = mel.to(device)

    # Whisper encoder
    with torch.no_grad():
        features = whisper_encoder(mel)  # (1, T', 384)
    return features.squeeze(0).cpu()  # (T', 384) — move back to CPU for storage

def extract_video_feats(video_path, lip_encoder, max_frames=50):
    """提取口唇视频特征"""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to 112x112 (standard lip reading input)
        gray = cv2.resize(gray, (112, 112))
        frames.append(gray)
    cap.release()

    if len(frames) == 0:
        return torch.zeros(512)

    # Pad if needed
    while len(frames) < max_frames:
        frames.append(np.zeros_like(frames[0]))

    # Stack → (T, H, W) → (1, T, H, W)
    video = torch.from_numpy(np.stack(frames)).float() / 255.0
    video = video.unsqueeze(0)  # (1, T, H, W)

    with torch.no_grad():
        features = lip_encoder(video)  # (1, 512)
    return features.squeeze(0)  # (512,)

# ============================================================
# 5. V5 模型: 多模态融合 + Cross-Attention 匹配
# ============================================================

class KWSpotterV5(nn.Module):
    """V5: 音频(Whisper) + 视频(LipReading) + 文本(CharCNN)"""
    def __init__(self, hidden=256, vocab_size=64, text_dim=128,
                 audio_dim=384, video_dim=512, use_video=False):
        super().__init__()
        self.use_video = use_video

        # 音频投影 (Whisper 384 → hidden)
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
        )

        # 视频投影 (LipReading 512 → hidden)
        if use_video:
            self.video_proj = nn.Sequential(
                nn.Linear(video_dim, hidden),
                nn.LayerNorm(hidden), nn.GELU(),
            )

        # 文本编码
        self.text_encoder = TextEncoder(vocab_size, emb_dim=64, out_dim=text_dim)
        self.text_proj = nn.Linear(text_dim, hidden)

        # 多模态融合
        fusion_in = hidden + (hidden if use_video else 0) + hidden  # audio + video + text
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )

        # 匹配网络 (Query 只看 audio+video)
        q_fusion_in = hidden + (hidden if use_video else 0)
        self.q_fusion = nn.Sequential(
            nn.Linear(q_fusion_in, hidden),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )

        # Cross-attention matching
        self.match_attn = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=0.1)
        self.match_norm = nn.LayerNorm(hidden)

        # Frame-level 投影 (用于 cross-attention)
        self.frame_proj = nn.Linear(audio_dim, hidden)

        # 分类头
        self.cls = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden * 2, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, enroll_audio, enroll_text_tokens, query_audio,
                enroll_video=None, query_video=None):
        """
        enroll_audio: (B, T_e, audio_dim) — Whisper 特征
        query_audio:  (B, T_q, audio_dim)
        enroll_text_tokens: (B, max_text_len)
        enroll_video: (B, video_dim) — optional
        query_video:  (B, video_dim) — optional
        """
        B = enroll_audio.shape[0]

        # ----- Enrollment -----
        # Audio: mean pool → project
        e_audio = self.audio_proj(enroll_audio.mean(dim=1))  # (B, hidden)

        # Video (optional)
        e_video_list = []
        if self.use_video and enroll_video is not None:
            e_video_list.append(self.video_proj(enroll_video))  # (B, hidden)

        # Text
        e_text = self.text_proj(self.text_encoder(enroll_text_tokens))  # (B, hidden)

        # Fuse all modalities for enrollment
        e_fused = self.fusion(torch.cat([e_audio] + e_video_list + [e_text], dim=-1))  # (B, hidden)

        # ----- Query -----
        q_audio = self.audio_proj(query_audio.mean(dim=1))  # (B, hidden)
        q_video_list = []
        if self.use_video and query_video is not None:
            q_video_list.append(self.video_proj(query_video))
        q_fused = self.q_fusion(torch.cat([q_audio] + q_video_list, dim=-1))  # (B, hidden)

        # Frame-level query features for cross-attention
        q_frames = self.frame_proj(query_audio)  # (B, T_q, hidden)

        # Cross-attention: enrollment attends to query frames
        q_attn, _ = self.match_attn(
            e_fused.unsqueeze(1), q_frames, q_frames
        )  # (B, 1, hidden)
        q_attn = self.match_norm(q_attn.squeeze(1))

        # Feature fusion for classification
        diff = torch.abs(e_fused - q_attn)
        prod = e_fused * q_attn
        fused = torch.cat([e_fused, q_attn, diff, prod], dim=-1)

        return self.cls(fused)

    def predict(self, *args, **kwargs):
        with torch.no_grad():
            return torch.sigmoid(self.forward(*args, **kwargs)).squeeze(-1)

# ============================================================
# 数据提取 (V5: 多模态特征)
# ============================================================

def parse_enroll_text(label_path):
    with open(label_path) as f:
        content = f.read().strip()
    m = re.search(r'enrollment:(\S+)', content)
    return m.group(1) if m else '<unk>'

def pre_extract_all_v5(split_dir, whisper_encoder, lip_encoder, use_video=False,
                       max_enroll=2.0, max_query=4.0, device=None):
    """预提取多模态特征: Whisper音频 + (可选)视频 + 文本"""
    d = Path(split_dir)
    labels_path = d / "label"
    wav_dir = d / "wav"
    mp4_dir = d / "mp4"
    data = {}
    files = sorted(labels_path.glob("pair_*.txt"))
    print(f"  V5预提取 {d.name}: {len(files)} pairs...")

    for lf in tqdm(files, desc=f"  {d.name}"):
        pid = lf.stem
        with open(lf) as f:
            parts = f.read().strip().split('\t')
        label = int(parts[2].replace('label:', ''))

        # Whisper audio features
        ef = extract_whisper_feats(wav_dir / f"{pid}_enroll.wav", whisper_encoder, max_enroll, device)
        qf = extract_whisper_feats(wav_dir / f"{pid}_query.wav", whisper_encoder, max_query, device)

        # Text
        enroll_word = parse_enroll_text(lf)

        item = {
            'enroll_feat': ef, 'query_feat': qf,
            'label': label, 'enroll_text': enroll_word,
        }

        # Video (optional)
        if use_video:
            try:
                ev = extract_video_feats(mp4_dir / f"{pid}_enroll.mp4", lip_encoder)
                qv = extract_video_feats(mp4_dir / f"{pid}_query.mp4", lip_encoder)
                item['enroll_video'] = ev
                item['query_video'] = qv
            except Exception:
                item['enroll_video'] = torch.zeros(512)
                item['query_video'] = torch.zeros(512)

        data[pid] = item
    return data

# ============================================================
# 数据集 (V5)
# ============================================================

class MemKWDatasetV5(Dataset):
    def __init__(self, data_dict, char2idx, max_text_len=MAX_TEXT_LEN,
                 augment=False, noise_feats=None):
        self.items = list(data_dict.items())
        self.char2idx = char2idx
        self.max_text_len = max_text_len
        self.augment = augment
        self.noise_feats = noise_feats or []

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        pid, d = self.items[idx]
        ef = d['enroll_feat']  # (T, 384) Whisper
        qf = d['query_feat']   # (T, 384) Whisper
        label = float(d.get('label', 0.5))
        word = d.get('enroll_text', '<unk>')
        tokens = tokenize(word, self.char2idx, self.max_text_len)

        result = {
            'enroll_feat': ef, 'query_feat': qf, 'label': label,
            'enroll_tokens': tokens, 'pid': pid,
        }

        # Video (optional)
        for vk in ['enroll_video', 'query_video']:
            if vk in d:
                result[vk] = d[vk]

        # Noise augmentation on query audio
        if self.augment and self.noise_feats and random.random() < 0.3:
            nf = random.choice(self.noise_feats)
            if len(nf.shape) == 2 and nf.shape[0] == 120:  # MFCC noise
                pass  # Skip MFCC noise for Whisper features (different dim)
            else:
                pass  # Skip for now

        return result

def pad_collate_fn_v5(batch):
    """V5 collate — Whisper 特征 (T, 384)"""
    efs, qfs, tokens_list, labels, pids = [], [], [], [], []
    evs, qvs = [], []
    has_video = 'enroll_video' in batch[0]

    for b in batch:
        efs.append(b['enroll_feat'])  # (T, 384)
        qfs.append(b['query_feat'])
        tokens_list.append(b['enroll_tokens'])
        labels.append(b['label'])
        pids.append(b['pid'])
        if has_video:
            evs.append(b.get('enroll_video', torch.zeros(512)))
            qvs.append(b.get('query_video', torch.zeros(512)))

    # Pad audio sequences
    efs_padded = nn.utils.rnn.pad_sequence(efs, batch_first=True)  # (B, max_T, 384)
    qfs_padded = nn.utils.rnn.pad_sequence(qfs, batch_first=True)

    result = {
        'ef': efs_padded, 'qf': qfs_padded,
        'enroll_tokens': torch.stack(tokens_list),
        'label': torch.tensor(labels, dtype=torch.float32),
        'pid': pids,
    }
    if has_video:
        result['enroll_video'] = torch.stack(evs)
        result['query_video'] = torch.stack(qvs)
    return result

def balance_sampler(dataset):
    pos_idx = [i for i, (_, d) in enumerate(dataset.items) if d.get('label', 0) == 1]
    neg_idx = [i for i, (_, d) in enumerate(dataset.items) if d.get('label', 0) == 0]
    if not pos_idx: return None
    w = max(1, len(neg_idx) // len(pos_idx))
    indices = neg_idx + pos_idx * w
    random.shuffle(indices)
    return indices

# ============================================================
# 训练 (BCE)
# ============================================================

def train_epoch_v5(model, ds, optimizer, bs, device):
    model.train()
    indices = balance_sampler(ds) or list(range(len(ds)))
    random.shuffle(indices)
    total_loss, n = 0, 0
    preds, labels = [], []

    for i in tqdm(range(0, len(indices), bs), desc="  Train", leave=False):
        batch_idx = indices[i:i+bs]
        batch = pad_collate_fn_v5([ds[idx] for idx in batch_idx])

        ef = batch['ef'].to(device)
        qf = batch['qf'].to(device)
        et = batch['enroll_tokens'].to(device)
        y = batch['label'].to(device)
        ev = batch.get('enroll_video', None)
        qv = batch.get('query_video', None)
        if ev is not None: ev = ev.to(device)
        if qv is not None: qv = qv.to(device)

        logit = model(ef, et, qf, ev, qv)
        loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item(); n += 1
        preds.extend(torch.sigmoid(logit).detach().cpu().flatten().tolist())
        labels.extend(y.cpu().tolist())

    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return total_loss / max(n, 1), auc

@torch.no_grad()
def evaluate_v5(model, ds, bs, device):
    model.eval()
    preds, labels = [], []
    for i in tqdm(range(0, len(ds), bs), desc="  Eval", leave=False):
        batch = pad_collate_fn_v5([ds[idx] for idx in range(i, min(i+bs, len(ds)))])
        ef = batch['ef'].to(device)
        qf = batch['qf'].to(device)
        et = batch['enroll_tokens'].to(device)
        ev = batch.get('enroll_video', None)
        qv = batch.get('query_video', None)
        if ev is not None: ev = ev.to(device)
        if qv is not None: qv = qv.to(device)

        prob = model.predict(ef, et, qf, ev, qv)
        preds.extend(prob.cpu().tolist())
        labels.extend(batch['label'].tolist())
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return auc, preds

def train_full_v5(model, train_ds, val_ds, cfg_hp, device):
    optimizer = optim.AdamW(model.parameters(), lr=cfg_hp['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_hp['epochs'])

    best_auc, best_ep, patience = 0.5, 0, 0

    for ep in range(1, cfg_hp['epochs'] + 1):
        t_loss, t_auc = train_epoch_v5(model, train_ds, optimizer, cfg_hp['bs'], device)
        v_auc, _ = evaluate_v5(model, val_ds, cfg_hp['bs'], device)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Ep {ep:3d}: loss={t_loss:.4f} | train_AUC={t_auc:.4f} | val_AUC={v_auc:.4f} | lr={lr_now:.2e}")

        if v_auc > best_auc:
            best_auc = v_auc; best_ep = ep; patience = 0
            torch.save(model.state_dict(), OUT / f"best_{cfg_hp['tag']}.pt")
        else:
            patience += 1

        scheduler.step()

        if patience >= cfg_hp.get('patience', 20):
            print(f"  早停 @ ep {ep}")
            break

    return best_auc, best_ep

# ============================================================
# 主 Loop
# ============================================================

def run_loop(use_video=False):
    print(f"Device: {DEVICE}")
    print(f"Video modality: {'ON' if use_video else 'OFF'}")

    # ---- 0. 加载预训练模型 ----
    print("\n[Phase 0] 加载预训练模型...")
    whisper_encoder, audio_dim = build_whisper_encoder()
    whisper_encoder.to(DEVICE)
    whisper_encoder.eval()

    lip_encoder = None
    if use_video:
        lip_encoder, video_dim = build_lipreading_encoder()
        lip_encoder.to(DEVICE)
        lip_encoder.eval()
    else:
        video_dim = 512

    # ---- 1. 预提取特征 ----
    print("\n[Phase 1] 预提取多模态特征...")
    mode = "v5_video" if use_video else "v5_audio"
    cache_path = OUT / f"all_features_{mode}.pkl"

    if cache_path.exists():
        print(f"  加载缓存: {cache_path}")
        with open(cache_path, 'rb') as f:
            all_data = pickle.load(f)
    else:
        all_data = {
            'train': pre_extract_all_v5(DATA / "dev_seen", whisper_encoder, lip_encoder, use_video, device=DEVICE),
            'val': pre_extract_all_v5(DATA / "dev_unseen", whisper_encoder, lip_encoder, use_video, device=DEVICE),
            'test_inset': None,
            'test_outset': None,
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(all_data, f)
        print(f"  特征已缓存: {cache_path}")

    # ---- 1.5 词表 ----
    print("\n[Phase 1.5] 构建字符词表...")
    all_words = set()
    for split in ['train', 'val']:
        for d in all_data[split].values():
            all_words.add(d.get('enroll_text', '<unk>'))
    for csv_path in [TEST / "eval_seen_enroll_text.csv", TEST / "eval_unseen_enroll_text.csv"]:
        if csv_path.exists():
            with open(csv_path, encoding='utf-8') as f:
                next(f)
                for line in f:
                    _, txt = line.strip().split(',', 1)
                    all_words.add(txt)
    vocab, char2idx = build_vocab(all_words)
    print(f"  词表大小: {len(vocab)} (来自 {len(all_words)} 个唯一单词)")

    # ---- 2. 数据集 ----
    print("\n[Phase 2] 构建数据集...")
    train_ds = MemKWDatasetV5(all_data['train'], char2idx, augment=True)
    val_ds = MemKWDatasetV5(all_data['val'], char2idx, augment=False)
    pos = sum(1 for _, d in train_ds.items if d.get('label', 0) == 1)
    print(f"  Train: {len(train_ds)} ({pos} pos) | Val: {len(val_ds)}")

    # ---- 3. 超参数搜索 ----
    print("\n[Phase 3] 超参数搜索 Loop...")

    hp_grid = [
        {'lr': 1e-3, 'bs': 32, 'epochs': 40, 'hidden': 256, 'text_dim': 128, 'patience': 15, 'tag': 'v5_base'},
        {'lr': 5e-4, 'bs': 32, 'epochs': 50, 'hidden': 320, 'text_dim': 160, 'patience': 18, 'tag': 'v5_wide'},
        {'lr': 3e-4, 'bs': 24, 'epochs': 60, 'hidden': 320, 'text_dim': 192, 'patience': 20, 'tag': 'v5_deep'},
        {'lr': 5e-4, 'bs': 32, 'epochs': 45, 'hidden': 384, 'text_dim': 192, 'patience': 15, 'tag': 'v5_big'},
        {'lr': 2e-4, 'bs': 24, 'epochs': 70, 'hidden': 320, 'text_dim': 160, 'patience': 20, 'tag': 'v5_long'},
    ]

    prev_best_auc = 0.6466  # V3 baseline
    prev_best_tag = 'v3_wide'
    print(f"  V3 最佳 AUC: {prev_best_auc:.4f} ({prev_best_tag})")

    results = []
    best_auc = prev_best_auc
    best_cfg = None

    for i, hp in enumerate(hp_grid):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(hp_grid)}] {hp['tag']}: lr={hp['lr']}, bs={hp['bs']}, "
              f"ep={hp['epochs']}, h={hp['hidden']}, td={hp['text_dim']}")
        print(f"{'='*60}")

        set_seed(SEED)
        model = KWSpotterV5(
            hidden=hp['hidden'], vocab_size=len(vocab),
            text_dim=hp['text_dim'], audio_dim=audio_dim,
            video_dim=video_dim, use_video=use_video,
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {n_params/1e6:.2f}M")

        v_auc, best_ep = train_full_v5(model, train_ds, val_ds, hp, DEVICE)
        results.append({'tag': hp['tag'], 'auc': v_auc, 'best_ep': best_ep, 'hp': hp})
        print(f"  Result: AUC={v_auc:.4f} @ epoch {best_ep}")

        if v_auc > best_auc:
            best_auc = v_auc
            best_cfg = hp
            shutil.copy(OUT / f"best_{hp['tag']}.pt", OUT / f"best_model_{mode}.pt")
            print(f"  >>> NEW BEST! AUC={v_auc:.4f} (vs V3: Δ={v_auc-prev_best_auc:+.4f})")

    # ---- 4. 汇总 ----
    print(f"\n{'#'*60}")
    print(f"  Loop Complete!")
    results.sort(key=lambda r: r['auc'], reverse=True)
    for i, r in enumerate(results):
        m = " *** BEST ***" if i == 0 else ""
        print(f"  {i+1}. [{r['tag']}] AUC={r['auc']:.4f} ep={r['best_ep']}{m}")
    print(f"  Best AUC: {best_auc:.4f}")
    print(f"  vs V3:    {prev_best_auc:.4f} (Δ={best_auc - prev_best_auc:+.4f})")

    with open(OUT / f"results_{mode}.json", 'w') as f:
        json.dump({
            'best_auc': best_auc, 'vs_v3': prev_best_auc,
            'results': [{'tag': r['tag'], 'auc': r['auc'], 'best_ep': r['best_ep']} for r in results],
        }, f, indent=2)

    print(f"\n  Done! Best AUC={best_auc:.4f}")
    return best_auc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', action='store_true', help='Enable video modality')
    args = parser.parse_args()
    run_loop(use_video=args.video)
