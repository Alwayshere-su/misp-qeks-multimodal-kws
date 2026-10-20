"""
MISP-QEKS Baseline V6 — MFCC音频 + LipReading视频 多模态
========================================================
基于 V3 的成功架构 (AUC=0.6466), 叠加预训练口唇视频特征。

关键设计 (V4/V5 教训):
1. 音频: 完整保留 V3 AudioEncoderV3 — MFCC(120) → CNN(4层, stride2↓16x) → attn_pool
2. 视频: LipReading 3D ResNet (预训练, 冻结) → 512-dim → Linear投影 → 256-dim
3. 不添加文本模态 (与音频冗余, 零信息增益)
4. 视频帧用线性插值到固定50帧 (而非zero-pad, 避免破坏3D Conv)
5. BCE Loss + balance_sampler (V3验证)
6. 总参数 ~3.5M (V3的~2M + 视频投影)
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
OUT = BASE / "output_v6"
SUB = BASE / "submission"
OUT.mkdir(exist_ok=True); SUB.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SR = 16000

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(SEED)

# ============================================================
# 音频编码器 (完全复用 V3)
# ============================================================

class AudioEncoderV3(nn.Module):
    def __init__(self, in_c=120, hidden=256, out_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_c, hidden, 7, 2, 3), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, 2, 2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, 2, 2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, 2, 2), nn.BatchNorm1d(hidden), nn.GELU(),
        )
        self.attn = nn.Conv1d(hidden, 1, 1)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.encoder(x)          # (B, hidden, T/16)
        w = self.attn(x).softmax(-1)  # (B, 1, T/16)
        global_e = (w * x).sum(-1)    # (B, hidden)
        global_e = self.proj(global_e)  # (B, out_dim)
        frames = x.transpose(1, 2)     # (B, T/16, hidden)
        frames = self.proj(frames)     # (B, T/16, out_dim)
        return global_e, frames

# ============================================================
# 视频编码器 (LipReading 3D ResNet, 预训练 + 冻结)
# ============================================================

class LipReadingEncoder(nn.Module):
    """加载预训练 LipReading 模型 (3D frontend + 2D ResNet + temporal pool)"""
    def __init__(self, ckpt_path):
        super().__init__()
        # 3D frontend
        self.frontend = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        # 2D ResNet18 (逐帧处理)
        self.resnet = ResNet2D(BasicBlock2D, [2, 2, 2, 2])
        # Temporal pool + projection
        self.fc = nn.Linear(512, 256)
        self.bnfc = nn.BatchNorm1d(256)

        # 加载权重
        ckpt = Path(ckpt_path) if isinstance(ckpt_path, str) else ckpt_path
        if ckpt.exists():
            sd = torch.load(ckpt, map_location='cpu', weights_only=False)
            our_sd = {}
            for k, v in sd.items():
                if k.startswith('video_frontend.video_frontend.'):
                    our_sd[k.replace('video_frontend.video_frontend.', 'frontend.')] = v
                elif k.startswith('video_frontend.resnet.'):
                    new_k = k.replace('video_frontend.resnet.', '')
                    if new_k.startswith('fc.') or new_k.startswith('bnfc.'):
                        our_sd[new_k] = v
                    else:
                        our_sd['resnet.' + new_k] = v
            missing, unexpected = self.load_state_dict(our_sd, strict=False)
            n_matched = len(our_sd) - len(missing)
            print(f"  LipReading loaded: {n_matched}/{len(sd)} keys matched"
                  + (f" ({len(missing)} missing)" if len(missing) > 0 else ""))

        # 冻结所有预训练权重
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        """x: (B, 1, T, H, W)"""
        B = x.shape[0]
        x = self.frontend(x)      # (B, 64, T, H', W')
        B, C, T, H, W = x.shape
        # Reshape: treat each frame independently
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, 64, H', W')
        x = self.resnet(x)        # (B*T, 512, H'', W'')
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (B*T, 512, 1, 1)
        x = x.view(B, T, 512)     # (B, T, 512)
        x = x.mean(dim=1)         # (B, 512) temporal average
        x = self.bnfc(self.fc(x)) # (B, 256)
        return x

class BasicBlock2D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ResNet2D(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return x

# ============================================================
# V6 关键词检索器: audio + video → cross-attention
# ============================================================

class KWSpotterV6(nn.Module):
    def __init__(self, hidden=256, use_video=True):
        super().__init__()
        self.use_video = use_video

        # 音频编码 (V3, 保留帧级信息)
        self.audio_encoder = AudioEncoderV3(120, hidden, hidden)

        # 视频投影
        if use_video:
            self.video_proj = nn.Sequential(
                nn.Linear(256, hidden),
                nn.BatchNorm1d(hidden), nn.GELU(),
            )

        # 多模态融合 (enrollment side)
        enroll_in = hidden + (hidden if use_video else 0)
        self.enroll_fuse = nn.Sequential(
            nn.Linear(enroll_in, hidden),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )

        # Query fusion
        query_in = hidden + (hidden if use_video else 0)
        self.query_fuse = nn.Sequential(
            nn.Linear(query_in, hidden),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )

        # 匹配网络
        self.match_attn = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=0.1)
        self.match_norm = nn.LayerNorm(hidden)

        # 分类头
        self.cls = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden * 2, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, ef, qf, ev=None, qv=None):
        """
        ef: (B, 120, Te) — MFCC enrollment
        qf: (B, 120, Tq) — MFCC query
        ev: (B, 256) — LipReading video enrollment (optional)
        qv: (B, 256) — LipReading video query (optional)
        """
        # Audio
        e_audio, e_frames = self.audio_encoder(ef)  # (B, H), (B, Te', H)
        q_audio, q_frames = self.audio_encoder(qf)  # (B, H), (B, Tq', H)

        # Video
        enroll_parts = [e_audio]
        query_parts = [q_audio]
        if self.use_video and ev is not None:
            enroll_parts.append(self.video_proj(ev))
        if self.use_video and qv is not None:
            query_parts.append(self.video_proj(qv))

        # Fuse
        e_fused = self.enroll_fuse(torch.cat(enroll_parts, dim=-1))  # (B, H)
        q_fused = self.query_fuse(torch.cat(query_parts, dim=-1))     # (B, H)

        # Cross-attention: enrollment → query frames
        q_attn, _ = self.match_attn(
            e_fused.unsqueeze(1), q_frames, q_frames
        )  # (B, 1, H)
        q_attn = self.match_norm(q_attn.squeeze(1))

        # Feature fusion
        diff = torch.abs(e_fused - q_attn)
        prod = e_fused * q_attn
        fused = torch.cat([e_fused, q_attn, diff, prod], dim=-1)

        return self.cls(fused)

    def predict(self, *args, **kwargs):
        with torch.no_grad():
            return torch.sigmoid(self.forward(*args, **kwargs)).squeeze(-1)

# ============================================================
# 音频特征提取 (V3 compatible)
# ============================================================

def extract_mfcc_feats(wav_path, max_sec):
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
    feat = torch.cat([mfcc, d1, d2], dim=1)
    return feat.squeeze(0)  # (120, T)

# ============================================================
# 视频特征提取 (LipReading, 插值到固定帧数)
# ============================================================

def extract_video_feats(video_path, lip_encoder, target_frames=50):
    """提取口唇视频特征: 读取 → 线性插值到target_frames → 过3D CNN → 返回256-dim"""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 96x96 已经是口唇ROI, 保持原分辨率
        frames.append(torch.from_numpy(gray).float() / 255.0)
    cap.release()

    if len(frames) == 0:
        return torch.zeros(256)

    # 线性插值到 target_frames
    frames_t = torch.stack(frames)  # (T, H, W)
    T = frames_t.shape[0]
    if T != target_frames:
        # 用线性插值缩放时间维度
        frames_t = frames_t.unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
        frames_t = F.interpolate(frames_t, size=(target_frames, 96, 96),
                                 mode='trilinear', align_corners=False)
        frames_t = frames_t.squeeze(0)  # (1, T, H, W)
    else:
        frames_t = frames_t.unsqueeze(0)  # (1, T, H, W)

    with torch.no_grad():
        features = lip_encoder(frames_t)  # (1, 256)
    return features.squeeze(0)  # (256,)

# ============================================================
# 数据预提取
# ============================================================

def pre_extract_all_v6(split_dir, lip_encoder, use_video,
                       max_enroll=2.0, max_query=4.0):
    """预提取 MFCC 音频 + (可选) LipReading 视频"""
    d = Path(split_dir)
    labels_path = d / "label"
    wav_dir = d / "wav"
    mp4_dir = d / "mp4"
    data = {}
    files = sorted(labels_path.glob("pair_*.txt"))
    print(f"  V6预提取 {d.name}: {len(files)} pairs...")

    for lf in tqdm(files, desc=f"  {d.name}"):
        pid = lf.stem
        with open(lf) as f:
            parts = f.read().strip().split('\t')
        label = int(parts[2].replace('label:', ''))

        # MFCC 音频
        ef = extract_mfcc_feats(wav_dir / f"{pid}_enroll.wav", max_enroll)
        qf = extract_mfcc_feats(wav_dir / f"{pid}_query.wav", max_query)

        item = {'enroll_feat': ef, 'query_feat': qf, 'label': label}

        # 视频 (可选)
        if use_video:
            try:
                ev = extract_video_feats(mp4_dir / f"{pid}_enroll.mp4", lip_encoder)
                qv = extract_video_feats(mp4_dir / f"{pid}_query.mp4", lip_encoder)
            except Exception:
                ev = torch.zeros(256)
                qv = torch.zeros(256)
            item['enroll_video'] = ev
            item['query_video'] = qv

        data[pid] = item
    return data

# ============================================================
# 数据集
# ============================================================

class MemKWDatasetV6(Dataset):
    def __init__(self, data_dict, augment=False, noise_feats=None):
        self.items = list(data_dict.items())
        self.augment = augment
        self.noise_feats = noise_feats or []

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        pid, d = self.items[idx]
        ef = d['enroll_feat']
        qf = d['query_feat']
        label = float(d.get('label', 0.5))
        result = {'enroll_feat': ef, 'query_feat': qf, 'label': label, 'pid': pid}

        # 噪声增强 (同V3)
        if self.augment and self.noise_feats and random.random() < 0.3:
            nf = random.choice(self.noise_feats)
            if nf.shape[1] < qf.shape[1]:
                reps = qf.shape[1] // nf.shape[1] + 1
                nf = nf.repeat(1, reps)
            nf = nf[:, :qf.shape[1]]
            snr_db = random.choice([-5, 0, 5])
            signal_pow = (qf**2).mean()
            noise_pow = (nf**2).mean()
            if noise_pow > 1e-10:
                k = (signal_pow / noise_pow) * (10**(-snr_db/10))
                k = max(0.01, min(k, 100))
                qf = qf + (k**0.5) * nf
            result['query_feat'] = qf

        # 视频 (可选)
        for vk in ['enroll_video', 'query_video']:
            if vk in d:
                result[vk] = d[vk]

        return result

def pad_collate_fn_v6(batch):
    efs, qfs, labels, pids = [], [], [], []
    evs, qvs = [], []
    has_video = 'enroll_video' in batch[0]

    for b in batch:
        efs.append(b['enroll_feat'].transpose(0, 1))
        qfs.append(b['query_feat'].transpose(0, 1))
        labels.append(b['label'])
        pids.append(b['pid'])
        if has_video:
            evs.append(b.get('enroll_video', torch.zeros(256)))
            qvs.append(b.get('query_video', torch.zeros(256)))

    efs = nn.utils.rnn.pad_sequence(efs, batch_first=True).transpose(1, 2)
    qfs = nn.utils.rnn.pad_sequence(qfs, batch_first=True).transpose(1, 2)

    result = {
        'ef': efs, 'qf': qfs,
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
# 训练
# ============================================================

def train_epoch_v6(model, ds, optimizer, bs, device):
    model.train()
    indices = balance_sampler(ds) or list(range(len(ds)))
    random.shuffle(indices)
    total_loss, n = 0, 0
    preds, labels = [], []

    has_video = 'enroll_video' in ds.items[0][1]

    for i in tqdm(range(0, len(indices), bs), desc="  Train", leave=False):
        batch_idx = indices[i:i+bs]
        batch = pad_collate_fn_v6([ds[idx] for idx in batch_idx])

        ef = batch['ef'].to(device)
        qf = batch['qf'].to(device)
        y = batch['label'].to(device)
        ev = batch.get('enroll_video', None)
        qv = batch.get('query_video', None)
        if ev is not None: ev = ev.to(device)
        if qv is not None: qv = qv.to(device)

        logit = model(ef, qf, ev, qv)
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
def evaluate_v6(model, ds, bs, device):
    model.eval()
    preds, labels = [], []
    has_video = 'enroll_video' in ds.items[0][1]

    for i in tqdm(range(0, len(ds), bs), desc="  Eval", leave=False):
        batch = pad_collate_fn_v6([ds[idx] for idx in range(i, min(i+bs, len(ds)))])
        ef = batch['ef'].to(device)
        qf = batch['qf'].to(device)
        ev = batch.get('enroll_video', None)
        qv = batch.get('query_video', None)
        if ev is not None: ev = ev.to(device)
        if qv is not None: qv = qv.to(device)

        prob = model.predict(ef, qf, ev, qv)
        preds.extend(prob.cpu().tolist())
        labels.extend(batch['label'].tolist())

    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return auc, preds

def train_full_v6(model, train_ds, val_ds, cfg_hp, device):
    optimizer = optim.AdamW(model.parameters(), lr=cfg_hp['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=8, factor=0.5)

    best_auc, best_ep, patience = 0.5, 0, 0

    for ep in range(1, cfg_hp['epochs'] + 1):
        t_loss, t_auc = train_epoch_v6(model, train_ds, optimizer, cfg_hp['bs'], device)
        v_auc, _ = evaluate_v6(model, val_ds, cfg_hp['bs'], device)
        scheduler.step(v_auc)

        print(f"  Ep {ep:3d}: loss={t_loss:.4f} | train_AUC={t_auc:.4f} | val_AUC={v_auc:.4f}")

        if v_auc > best_auc:
            best_auc = v_auc; best_ep = ep; patience = 0
            torch.save(model.state_dict(), OUT / f"best_{cfg_hp['tag']}.pt")
        else:
            patience += 1

        if patience >= cfg_hp.get('patience', 12):
            print(f"  早停 @ ep {ep}")
            break

    return best_auc, best_ep

# ============================================================
# 主 Loop
# ============================================================

def run_loop(use_video=True):
    mode = "v6_video" if use_video else "v6_audio"
    print(f"Device: {DEVICE} | Video: {'ON' if use_video else 'OFF'}")

    # ---- 0. 加载视频编码器 ----
    lip_encoder = None
    if use_video:
        print("\n[Phase 0] 加载 LipReading 视频编码器...")
        ckpt_path = AUX / "pretrained_feature_models/lipreading/lipreading_LRW_0.8018.pt"
        lip_encoder = LipReadingEncoder(ckpt_path)
        lip_encoder.to(DEVICE)
        lip_encoder.eval()
        print(f"  LipReading params: {sum(p.numel() for p in lip_encoder.parameters())/1e6:.1f}M")

    # ---- 1. 预提取特征 ----
    print("\n[Phase 1] 预提取 MFCC + 视频特征...")
    cache_path = OUT / f"all_features_{mode}.pkl"

    if cache_path.exists():
        print(f"  加载缓存: {cache_path}")
        with open(cache_path, 'rb') as f:
            all_data = pickle.load(f)
    else:
        all_data = {
            'train': pre_extract_all_v6(DATA / "dev_seen", lip_encoder, use_video),
            'val': pre_extract_all_v6(DATA / "dev_unseen", lip_encoder, use_video),
            'test_inset': None,
            'test_outset': None,
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(all_data, f)
        print(f"  特征已缓存: {cache_path}")

    # ---- 1.5 噪声 ----
    noise_feats = []
    noise_pkl = OUT / "noise_feats_v6.pkl"
    if noise_pkl.exists():
        with open(noise_pkl, 'rb') as f:
            noise_feats = pickle.load(f)
    else:
        try:
            noise_dir = AUX / "noise"
            for nd in noise_dir.iterdir():
                if nd.is_dir():
                    for wf in list(nd.glob("*.wav"))[:50]:
                        try:
                            nf = extract_mfcc_feats(wf, max_sec=4.0)
                            noise_feats.append(nf)
                        except: pass
            with open(noise_pkl, 'wb') as f:
                pickle.dump(noise_feats, f)
        except Exception as e:
            print(f"  噪声跳过: {e}")
    print(f"  噪声特征: {len(noise_feats)}")

    # ---- 2. 数据集 ----
    print("\n[Phase 2] 构建数据集...")
    train_ds = MemKWDatasetV6(all_data['train'], augment=True, noise_feats=noise_feats)
    val_ds = MemKWDatasetV6(all_data['val'], augment=False)
    pos = sum(1 for _, d in train_ds.items if d.get('label', 0) == 1)
    print(f"  Train: {len(train_ds)} ({pos} pos) | Val: {len(val_ds)}")

    # ---- 3. 超参搜索 ----
    print("\n[Phase 3] 超参数搜索 Loop...")

    hp_grid = [
        {'lr': 1e-3, 'bs': 64,  'epochs': 50, 'hidden': 256, 'patience': 12, 'tag': 'v6_base'},
        {'lr': 5e-4, 'bs': 64,  'epochs': 60, 'hidden': 320, 'patience': 15, 'tag': 'v6_wide'},
        {'lr': 3e-4, 'bs': 48,  'epochs': 50, 'hidden': 320, 'patience': 12, 'tag': 'v6_video'},
        {'lr': 2e-4, 'bs': 48,  'epochs': 70, 'hidden': 384, 'patience': 18, 'tag': 'v6_deep'},
        {'lr': 3e-4, 'bs': 64,  'epochs': 80, 'hidden': 320, 'patience': 18, 'tag': 'v6_long'},
    ]

    prev_best_auc = 0.6466  # V3 baseline
    prev_best_tag = 'v3_wide'
    print(f"  V3 baseline AUC: {prev_best_auc:.4f} ({prev_best_tag})")

    results = []
    best_auc = prev_best_auc
    best_cfg = None

    for i, hp in enumerate(hp_grid):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(hp_grid)}] {hp['tag']}: lr={hp['lr']}, bs={hp['bs']}, "
              f"ep={hp['epochs']}, h={hp['hidden']}")
        print(f"{'='*60}")

        set_seed(SEED)
        model = KWSpotterV6(hidden=hp['hidden'], use_video=use_video).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  可训练参数: {n_params/1e6:.2f}M")

        v_auc, best_ep = train_full_v6(model, train_ds, val_ds, hp, DEVICE)
        results.append({'tag': hp['tag'], 'auc': v_auc, 'best_ep': best_ep, 'hp': hp})
        print(f"  Result: AUC={v_auc:.4f} @ epoch {best_ep}")

        if v_auc > best_auc:
            best_auc = v_auc
            best_cfg = hp
            shutil.copy(OUT / f"best_{hp['tag']}.pt", OUT / f"best_model_{mode}.pt")
            delta = v_auc - prev_best_auc
            print(f"  >>> NEW BEST! AUC={v_auc:.4f} (vs V3: {'+' if delta>0 else ''}{delta:+.4f})")

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

    if best_auc > prev_best_auc:
        print(f"\n  🎉 超越V3! 生成提交文件...")
        # 提交生成逻辑 (后续补充)
    else:
        print(f"\n  V3仍是最佳 ({prev_best_auc:.4f})")

    return best_auc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--novideo', action='store_true')
    args = parser.parse_args()
    run_loop(use_video=not args.novideo)
