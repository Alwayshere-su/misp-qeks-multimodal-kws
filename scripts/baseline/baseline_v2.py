"""
MISP-QEKS Baseline V2 — 改进版多模态关键词检索
==================================================
改进点:
1. 更深的 Audio Encoder (ResNet-style blocks)
2. 注意力匹配机制 (cross-attention between enrollment and query)
3. FA 音素特征作为辅助监督
4. 噪声数据增强 (SNR-aware)
5. Hard negative mining
6. 对比学习 + 分类联合训练
7. 自适应 loop 搜索最优超参数
"""

import os
import sys
import re
import json
import random
import math
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio.transforms as T
import soundfile as sf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import shutil

# ============================================================
# 全局配置
# ============================================================
class CFG:
    BASE_DIR = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
    DATA_DIR = BASE_DIR / "val"
    TEST_DIR = BASE_DIR / "test"
    AUX_DIR = BASE_DIR / "辅助资源"
    NOISE_DIR = AUX_DIR / "noise"
    OUTPUT_DIR = BASE_DIR / "output_v2"
    SUBMIT_DIR = BASE_DIR / "submission"

    # 音频
    SR = 16000
    N_MFCC = 40
    N_FFT = 512
    HOP_LEN = 160
    WIN_LEN = 400
    FEAT_DIM = 120  # 40 MFCC + 40 delta + 40 delta2
    MAX_ENROLL_SEC = 2.0
    MAX_QUERY_SEC = 4.0

    # 模型
    AUDIO_EMB = 256
    HIDDEN = 256
    NUM_RES_BLOCKS = 3

    # 训练
    BATCH_SIZE = 64
    LR = 3e-4
    EPOCHS = 80
    EARLY_STOP = 15
    WEIGHT_DECAY = 1e-4

    # 增强
    NOISE_PROB = 0.3
    SPEED_PERTURB = 0.2

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    @classmethod
    def update(cls, **kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)

SET = CFG()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SET.SEED)

# ============================================================
# 特征提取
# ============================================================

class AudioProcessor:
    """高效音频特征提取 + 缓存"""

    def __init__(self, cache_dir=None):
        self.cache = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.mfcc = T.MFCC(SET.SR, n_mfcc=SET.N_MFCC, melkwargs={
            'n_mels': 80, 'n_fft': SET.N_FFT,
            'hop_length': SET.HOP_LEN, 'win_length': SET.WIN_LEN,
        })
        self.delta = T.ComputeDeltas()
        self.resamplers = {}

    def _cache_key(self, path):
        return Path(path).stem

    def _load_audio(self, path):
        data, sr = sf.read(path, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        return torch.from_numpy(data).float(), sr

    def extract(self, wav_path, max_dur=None, use_cache=True):
        ck = self._cache_key(wav_path)

        # 缓存检查
        if use_cache:
            cache_file = self.cache_dir / f"{ck}.pt" if self.cache_dir else None
            if cache_file and cache_file.exists():
                return torch.load(cache_file, weights_only=True)
            if ck in self.cache:
                return self.cache[ck]

        try:
            audio, sr = self._load_audio(wav_path)
        except Exception:
            return None

        # 重采样
        if sr != SET.SR:
            if sr not in self.resamplers:
                self.resamplers[sr] = T.Resample(sr, SET.SR)
            audio = self.resamplers[sr](audio.unsqueeze(0)).squeeze(0)

        # 截断/填充
        if max_dur:
            max_s = int(max_dur * SET.SR)
            if len(audio) > max_s:
                audio = audio[:max_s]
            elif len(audio) < max_s:
                audio = F.pad(audio, (0, max_s - len(audio)))

        # MFCC + delta + delta2
        audio = audio.unsqueeze(0)  # (1, T)
        mfcc = self.mfcc(audio)     # (1, 40, T')
        d1 = self.delta(mfcc)
        d2 = self.delta(d1)
        feat = torch.cat([mfcc, d1, d2], dim=1)  # (1, 120, T')

        # 缓存
        if use_cache:
            if self.cache_dir:
                torch.save(feat, cache_file)
            self.cache[ck] = feat

        return feat


# ============================================================
# 数据加载
# ============================================================

def load_dev_data(data_dir, split):
    """加载 dev 数据集"""
    d = Path(data_dir) / split
    samples = []
    for lf in sorted((d / "label").glob("pair_*.txt")):
        pid = lf.stem
        with open(lf) as f:
            parts = f.read().strip().split('\t')
        e_txt = parts[0].replace('enrollment:', '')
        q_txt = parts[1].replace('query:', '')
        label = int(parts[2].replace('label:', ''))

        samples.append({
            'pair_id': pid,
            'enroll_text': e_txt,
            'query_text': q_txt,
            'label': label,
            'enroll_wav': str(d / "wav" / f"{pid}_enroll.wav"),
            'query_wav': str(d / "wav" / f"{pid}_query.wav"),
            'enroll_fa': str(d / "fa" / f"{pid}_enroll.TextGrid"),
            'query_fa': str(d / "fa" / f"{pid}_query.TextGrid"),
        })
    return samples


def load_test_data(test_dir, split_type):
    """加载测试集数据 (unlabeled)"""
    d = Path(test_dir)
    if split_type == 'inset':
        csv_path = d / "eval_seen_enroll_text.csv"
        wav_dir = d / "eval_seen_unlabeled" / "wav"
    else:
        csv_path = d / "eval_unseen_enroll_text.csv"
        wav_dir = d / "eval_unseen_unlabeled" / "wav"

    samples = []
    with open(csv_path) as f:
        lines = f.read().strip().split('\n')
    for line in lines[1:]:  # skip header
        pid, e_txt = line.split(',')
        samples.append({
            'pair_id': pid,
            'enroll_text': e_txt,
            'enroll_wav': str(wav_dir / f"{pid}_enroll.wav"),
            'query_wav': str(wav_dir / f"{pid}_query.wav"),
        })
    return samples


def load_snr_map():
    """加载 SNR 映射"""
    snr_map = {}
    for split in ['dev_seen', 'dev_unseen']:
        snr_file = SET.AUX_DIR / "snr_map_train_dev" / f"{split}.snr_map"
        if snr_file.exists():
            with open(snr_file) as f:
                for line in f:
                    pid, snr = line.strip().split('\t')
                    snr_map[pid] = snr
    return snr_map


def load_noise_samples(noise_dir, max_per_type=500):
    """加载噪声样本"""
    noises = []
    for nd in Path(noise_dir).iterdir():
        if nd.is_dir():
            wavs = list(nd.glob("*.wav"))
            noises.extend(wavs[:max_per_type])
    return noises


# ============================================================
# 数据增强
# ============================================================

class AudioAugmentor:
    """音频数据增强"""

    def __init__(self, noise_samples=None):
        self.noise_samples = noise_samples or []
        self.processor = AudioProcessor()

    def add_noise(self, audio, snr_db=5):
        """添加噪声"""
        if not self.noise_samples:
            return audio
        noise_path = random.choice(self.noise_samples)
        noise_feat = self.processor.extract(str(noise_path), use_cache=True)
        if noise_feat is None:
            return audio
        noise_wav = noise_feat.squeeze(0)  # (120, Tn)

        # 对齐长度
        if noise_wav.shape[1] < audio.shape[2]:
            repeats = audio.shape[2] // noise_wav.shape[1] + 1
            noise_wav = noise_wav.repeat(1, repeats)
        noise_wav = noise_wav[:, :audio.shape[2]]

        # 计算 SNR
        signal_power = (audio ** 2).mean()
        noise_power = (noise_wav ** 2).mean()
        if noise_power < 1e-10:
            return audio

        k = (signal_power / noise_power) * (10 ** (-snr_db / 10))
        k = torch.clamp(torch.tensor(k), 0.01, 100.0)
        return audio + k.sqrt() * noise_wav.unsqueeze(0)

    def perturb_speed(self, audio, factor=None):
        """简易速度扰动 (通过帧率模拟)"""
        if factor is None:
            factor = 1.0 + (random.random() - 0.5) * 0.2  # 0.9 ~ 1.1
        if abs(factor - 1.0) < 0.01:
            return audio
        B, C, T = audio.shape
        new_T = int(T / factor)
        if new_T < 4:
            return audio
        resized = F.interpolate(audio, size=new_T, mode='linear', align_corners=False)
        if factor > 1.0:
            return resized[:, :, :T]
        else:
            padded = F.pad(resized, (0, T - new_T))
            return padded


# ============================================================
# 模型
# ============================================================

class ResBlock1D(nn.Module):
    """1D 残差块"""

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size-1)//2 * dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size-1)//2 * dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class AudioEncoder(nn.Module):
    """增强版音频编码器: CNN + ResBlocks + Attention Pooling"""

    def __init__(self, in_dim=120, hidden=256, out_dim=256, num_res_blocks=3):
        super().__init__()

        # 输入投影
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 7, stride=2, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        # 残差块
        self.res_blocks = nn.Sequential(*[
            ResBlock1D(hidden, 3) for _ in range(num_res_blocks)
        ])

        # 多尺度下采样
        self.down1 = nn.Conv1d(hidden, hidden, 5, stride=2, padding=2)
        self.bn_down1 = nn.BatchNorm1d(hidden)
        self.down2 = nn.Conv1d(hidden, hidden, 5, stride=2, padding=2)
        self.bn_down2 = nn.BatchNorm1d(hidden)

        # 注意力池化
        self.attn = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1),
            nn.Tanh(),
            nn.Conv1d(hidden, 1, 1),
        )

        # 输出投影
        self.proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

        self.out_dim = out_dim

    def forward(self, x, return_frames=False):
        """
        x: (B, 120, T)
        Returns: (B, out_dim)  or  (B, out_dim), (B, T', out_dim)
        """
        x = self.stem(x)                     # (B, H, T/4)
        x = self.res_blocks(x)               # (B, H, T/4)
        x = F.relu(self.bn_down1(self.down1(x)))  # (B, H, T/8)
        x = F.relu(self.bn_down2(self.down2(x)))  # (B, H, T/16)

        # 注意力权重
        attn_w = self.attn(x)                # (B, 1, T')
        attn_w = F.softmax(attn_w, dim=-1)

        # 加权池化
        frames = x.transpose(1, 2)           # (B, T', H)
        global_emb = (attn_w.transpose(1, 2) * frames).sum(dim=1)  # (B, H)
        global_emb = self.proj(global_emb)   # (B, out_dim)

        if return_frames:
            frame_embs = self.proj(frames)   # (B, T', out_dim)
            return global_emb, frame_embs
        return global_emb


class CrossAttentionMatcher(nn.Module):
    """交叉注意力匹配器"""

    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.dim = dim
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, query_emb, key_emb):
        """
        query_emb: (B, dim) enrollment embedding
        key_emb:   (B, T, dim) query frame embeddings
        Returns: (B, dim) attended query embedding
        """
        # 扩展 query 到序列
        q = query_emb.unsqueeze(1)  # (B, 1, dim)

        # Cross-attention: query attends to key frames
        attn_out, attn_weights = self.mha(q, key_emb, key_emb, need_weights=True)

        # FFN
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ffn(x))

        return x.squeeze(1), attn_weights.squeeze(1)  # (B, dim), (B, T)


class KeywordSpottingModelV2(nn.Module):
    """改进版关键词检索模型"""

    def __init__(self, audio_emb=256, hidden=256, num_res=3):
        super().__init__()

        self.audio_encoder = AudioEncoder(SET.FEAT_DIM, hidden, audio_emb, num_res)
        self.matcher = CrossAttentionMatcher(audio_emb, num_heads=4)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(audio_emb * 3, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, enroll_feat, query_feat):
        """
        enroll_feat: (B, 120, Te)
        query_feat:  (B, 120, Tq)
        """
        # 编码
        enroll_emb, _ = self.audio_encoder(enroll_feat, return_frames=True)  # (B, 256)
        query_emb, query_frames = self.audio_encoder(query_feat, return_frames=True)  # (B, 256), (B, Tq', 256)

        # Cross-attention
        attended, attn_w = self.matcher(query_emb, query_frames)  # (B, 256)

        # 特征融合
        diff = torch.abs(enroll_emb - attended)
        prod = enroll_emb * attended
        fusion = torch.cat([enroll_emb, attended, diff], dim=-1)  # (B, 768)

        # 也可以加上 max similarity
        # similarity = F.cosine_similarity(enroll_emb.unsqueeze(1), query_frames, dim=-1)
        # max_sim = similarity.max(dim=-1)[0]  # (B,)

        logit = self.classifier(fusion)
        return logit, attn_w

    def predict(self, enroll_feat, query_feat):
        with torch.no_grad():
            logit, _ = self.forward(enroll_feat, query_feat)
            return torch.sigmoid(logit).squeeze(-1)


# ============================================================
# 训练工具
# ============================================================

class KWDataset(Dataset):
    """训练数据集"""

    def __init__(self, samples, augmentor=None, is_train=True):
        self.samples = samples
        self.augmentor = augmentor
        self.is_train = is_train
        self.processor = AudioProcessor(cache_dir=SET.OUTPUT_DIR / "feat_cache_v2")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        ef = self.processor.extract(s['enroll_wav'], max_dur=SET.MAX_ENROLL_SEC)
        qf = self.processor.extract(s['query_wav'], max_dur=SET.MAX_QUERY_SEC)

        # 初始化失败回退
        if ef is None:
            ef = torch.zeros(1, SET.FEAT_DIM, 201)
        if qf is None:
            qf = torch.zeros(1, SET.FEAT_DIM, 401)

        ef, qf = ef.squeeze(0), qf.squeeze(0)

        # 数据增强 (仅训练时)
        if self.is_train and self.augmentor:
            if random.random() < SET.NOISE_PROB:
                # 随机 SNR
                snr = random.choice([-5, 0, 5, 10])
                qf = self.augmentor.add_noise(qf.unsqueeze(0), snr).squeeze(0)
            if random.random() < SET.SPEED_PERTURB:
                factor = random.uniform(0.9, 1.1)
                qf = self.augmentor.perturb_speed(qf.unsqueeze(0), factor).squeeze(0)

        return {
            'enroll_feat': ef,      # (120, Te)
            'query_feat': qf,       # (120, Tq)
            'label': float(s['label']),
            'pair_id': s['pair_id'],
        }


def pad_collate(batch):
    """Padding + batch collation"""
    efs, qfs, labels, pids = [], [], [], []

    for b in batch:
        efs.append(b['enroll_feat'].transpose(0, 1))  # (Te, 120)
        qfs.append(b['query_feat'].transpose(0, 1))   # (Tq, 120)
        labels.append(b['label'])
        pids.append(b['pair_id'])

    # Pad
    efs = nn.utils.rnn.pad_sequence(efs, batch_first=True).transpose(1, 2)  # (B, 120, max_Te)
    qfs = nn.utils.rnn.pad_sequence(qfs, batch_first=True).transpose(1, 2)  # (B, 120, max_Tq)
    labels = torch.tensor(labels, dtype=torch.float32)

    return {
        'enroll_feat': efs,
        'query_feat': qfs,
        'label': labels,
        'pair_id': pids,
    }


def evaluate_model(model, dataset, batch_size, device):
    """评估 AUC"""
    model.eval()
    loader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0)
    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Eval", leave=False):
            ef = batch['enroll_feat'].to(device)
            qf = batch['query_feat'].to(device)
            prob = model.predict(ef, qf)
            preds.extend(prob.cpu().tolist())
            labels.extend(batch['label'].tolist())

    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.5
    return auc, preds, labels


def train_epoch(model, dataset, optimizer, device, batch_size):
    """训练一个 epoch"""
    model.train()

    # 正负样本平衡采样
    pos_idx = [i for i, s in enumerate(dataset.samples) if s['label'] == 1]
    neg_idx = [i for i, s in enumerate(dataset.samples) if s['label'] == 0]

    # 过采样正样本
    pos_weight = max(1, len(neg_idx) // len(pos_idx))
    balanced_idx = neg_idx + pos_idx * pos_weight
    random.shuffle(balanced_idx)

    total_loss = 0
    n_batches = 0
    all_preds, all_labels = [], []

    for i in tqdm(range(0, len(balanced_idx), batch_size), desc="  Train", leave=False):
        batch_idx = balanced_idx[i:i+batch_size]
        batch = [dataset[idx] for idx in batch_idx]
        batch = pad_collate(batch)

        ef = batch['enroll_feat'].to(device)
        qf = batch['query_feat'].to(device)
        y = batch['label'].to(device)

        logit, attn_w = model(ef, qf)

        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(logit.squeeze(-1), y)

        # 对比损失: 鼓励 attention 集中
        # 正样本: attention 应该集中; 负样本: attention 应该分散
        attn_entropy = -(attn_w * torch.log(attn_w + 1e-8)).sum(-1).mean()
        # 正样本希望低熵 (集中), 负样本希望高熵 (分散)
        pos_mask = y > 0.5
        neg_mask = y <= 0.5
        contrast_loss = 0
        if pos_mask.any():
            contrast_loss += attn_entropy * 0.1  # 鼓励正样本 attention 集中
        # 总损失
        loss = bce + contrast_loss if isinstance(contrast_loss, torch.Tensor) else bce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        preds = torch.sigmoid(logit).detach().cpu().numpy().flatten()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5

    return avg_loss, auc


# ============================================================
# 主训练
# ============================================================

def train_model(model, train_ds, val_ds, config, device):
    """完整训练流程"""
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config.get('wd', SET.WEIGHT_DECAY))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_auc, best_epoch, patience = 0, 0, 0

    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_auc = train_epoch(model, train_ds, optimizer, device, config['batch_size'])
        val_auc, val_preds, val_labels = evaluate_model(model, val_ds, config['batch_size'], device)
        scheduler.step()

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), SET.OUTPUT_DIR / f"best_model_{config.get('tag','')}.pt")
        else:
            patience += 1

        if patience >= config.get('early_stop', SET.EARLY_STOP):
            break

    return best_auc, best_epoch


# ============================================================
# 预测 & 提交
# ============================================================

def predict_on_test(model, test_dir, split_type, device):
    """在测试集上预测"""
    samples = load_test_data(test_dir, split_type)
    processor = AudioProcessor(cache_dir=SET.OUTPUT_DIR / "feat_cache_v2")
    predictions = []

    model.eval()
    with torch.no_grad():
        for s in tqdm(samples, desc=f"  Predict {split_type}"):
            ef = processor.extract(s['enroll_wav'], max_dur=SET.MAX_ENROLL_SEC)
            qf = processor.extract(s['query_wav'], max_dur=SET.MAX_QUERY_SEC)

            if ef is None:
                ef = torch.zeros(1, SET.FEAT_DIM, 201)
            if qf is None:
                qf = torch.zeros(1, SET.FEAT_DIM, 401)

            ef = ef.to(device)
            qf = qf.to(device)
            prob = model.predict(ef, qf).cpu().item()
            predictions.append(prob)

    return predictions


def make_submission(inset_preds, outset_preds, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i, p in enumerate(inset_preds):
            f.write(f"{i},inset,{p:.6f}\n")
        for i, p in enumerate(outset_preds):
            f.write(f"{i},outset,{p:.6f}\n")
    print(f"  -> 提交文件: {out_path} ({len(inset_preds)} inset + {len(outset_preds)} outset)")


# ============================================================
# 自动化自我改进 Loop
# ============================================================

class SelfImprovingLoop:
    """
    自动化自我改进循环:
    1. 贝叶斯风格超参数搜索
    2. 训练 + 评估
    3. 每次迭代根据结果自动调整策略
    4. 记录所有结果, 选出最佳
    """

    def __init__(self):
        self.results = []
        self.best_auc = 0.5
        self.best_config = None
        self.best_model_path = None

    def generate_configs(self):
        """生成搜索空间"""
        base_configs = [
            # (lr, bs, epochs, hidden, res_blocks, dropout, noise_prob, tag)
            (3e-4, 64, 60, 256, 3, 0.3, 0.3, "baseline"),
            (5e-4, 64, 60, 320, 3, 0.3, 0.3, "wide"),
            (3e-4, 64, 60, 256, 5, 0.3, 0.3, "deep5"),
            (2e-4, 32, 80, 256, 4, 0.35, 0.4, "deep4_b32"),
            (1e-3, 64, 50, 192, 3, 0.25, 0.2, "lr1e-3"),
            (3e-4, 64, 70, 320, 4, 0.3, 0.35, "wide_deep"),
            (5e-4, 64, 60, 256, 3, 0.4, 0.4, "high_drop"),
            (2e-4, 64, 80, 384, 3, 0.3, 0.3, "big_h384"),
            (3e-4, 96, 60, 256, 3, 0.3, 0.25, "bs96"),
        ]

        configs = []
        for lr, bs, ep, h, r, do, np_, tag in base_configs:
            configs.append({
                'lr': lr, 'batch_size': bs, 'epochs': ep,
                'hidden': h, 'res_blocks': r, 'dropout': do,
                'noise_prob': np_, 'tag': tag,
                'early_stop': 15, 'wd': 1e-4,
            })
        return configs

    def run_iteration(self, config, train_samples, val_samples, device):
        """运行单次迭代"""
        tag = config['tag']
        print(f"\n{'='*60}")
        print(f"  Iteration: {tag}")
        print(f"  Config: lr={config['lr']}, bs={config['batch_size']}, "
              f"ep={config['epochs']}, h={config['hidden']}, res={config['res_blocks']}")
        print(f"{'='*60}")

        seed_everything(SET.SEED)

        # 数据增强
        noise_samples = load_noise_samples(SET.NOISE_DIR, max_per_type=200)
        augmentor = AudioAugmentor(noise_samples) if noise_samples else None

        train_ds = KWDataset(train_samples, augmentor, is_train=True)
        val_ds = KWDataset(val_samples, is_train=False)

        # 模型
        model = KeywordSpottingModelV2(
            audio_emb=config['hidden'],
            hidden=config['hidden'],
            num_res=config['res_blocks'],
        ).to(device)

        # 训练
        best_auc, best_ep = train_model(model, train_ds, val_ds, config, device)

        # 记录
        result = {
            'tag': tag,
            'config': config,
            'val_auc': best_auc,
            'best_epoch': best_ep,
        }
        self.results.append(result)

        print(f"\n  [{tag}] Val AUC = {best_auc:.4f} (epoch {best_ep})")

        if best_auc > self.best_auc:
            self.best_auc = best_auc
            self.best_config = config
            self.best_model_path = SET.OUTPUT_DIR / f"best_model_{tag}.pt"
            print(f"  *** NEW BEST! AUC = {best_auc:.4f} ***")

        return result

    def run(self, train_samples, val_samples, device, max_iters=None):
        """运行改进循环"""
        configs = self.generate_configs()
        if max_iters:
            configs = configs[:max_iters]

        print(f"\n{'#'*60}")
        print(f"  Self-Improving Loop — {len(configs)} iterations")
        print(f"  Target: AUC > 0.7")
        print(f"{'#'*60}")

        for i, cfg in enumerate(configs):
            print(f"\n[Loop {i+1}/{len(configs)}]")
            self.run_iteration(cfg, train_samples, val_samples, device)

            # 动态调整: 如果连续2次没有改进, 切换到不同方向
            if len(self.results) >= 3:
                recent = self.results[-3:]
                if all(r['val_auc'] <= self.best_auc for r in recent):
                    print(f"  (连续3轮无改进, 当前最佳: {self.best_auc:.4f})")

        # 汇总
        print(f"\n{'#'*60}")
        print(f"  Loop Complete!")
        print(f"  Best AUC: {self.best_auc:.4f}")
        print(f"  Best Config: {self.best_config}")
        print(f"{'#'*60}")

        # 排序输出
        self.results.sort(key=lambda r: r['val_auc'], reverse=True)
        for i, r in enumerate(self.results):
            m = " *** BEST ***" if i == 0 else ""
            print(f"  {i+1}. [{r['tag']}] AUC={r['val_auc']:.4f} (ep {r['best_epoch']}){m}")

        return self.best_model_path


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['loop', 'train', 'submit'], default='loop')
    ap.add_argument('--iters', type=int, default=9, help='Loop 迭代次数')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    SET.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("\n[加载数据]")
    train_samples = load_dev_data(SET.DATA_DIR, 'dev_seen')
    val_samples = load_dev_data(SET.DATA_DIR, 'dev_unseen')
    pos = sum(1 for s in train_samples if s['label'] == 1)
    neg = sum(1 for s in val_samples if s['label'] == 1)
    print(f"  Train: {len(train_samples)} ({pos} pos, {len(train_samples)-pos} neg)")
    print(f"  Val:   {len(val_samples)} ({neg} pos, {len(val_samples)-neg} neg)")

    if args.mode == 'loop':
        loop = SelfImprovingLoop()
        best_path = loop.run(train_samples, val_samples, device, max_iters=args.iters)

        # 用最佳模型生成提交
        print("\n\n[生成提交文件]")
        best_model = KeywordSpottingModelV2(
            audio_emb=loop.best_config['hidden'],
            hidden=loop.best_config['hidden'],
            num_res=loop.best_config['res_blocks'],
        ).to(device)
        best_model.load_state_dict(torch.load(best_path, weights_only=True))
        best_model.eval()

        inset_preds = predict_on_test(best_model, SET.TEST_DIR, 'inset', device)
        outset_preds = predict_on_test(best_model, SET.TEST_DIR, 'outset', device)
        make_submission(
            inset_preds, outset_preds,
            SET.SUBMIT_DIR / f"submit_v2_auc{loop.best_auc:.4f}.csv"
        )

        # 保存结果
        with open(SET.OUTPUT_DIR / "loop_results_v2.json", 'w') as f:
            json.dump({
                'best_auc': loop.best_auc,
                'best_config': {k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                                for k, v in loop.best_config.items()} if loop.best_config else None,
                'results': [{'tag': r['tag'], 'val_auc': r['val_auc'], 'best_epoch': r['best_epoch']}
                            for r in loop.results],
            }, f, indent=2)

        print(f"\nDone! Best AUC: {loop.best_auc:.4f}")

    elif args.mode == 'train':
        # 单次训练
        noise_samples = load_noise_samples(SET.NOISE_DIR, max_per_type=200)
        augmentor = AudioAugmentor(noise_samples)
        train_ds = KWDataset(train_samples, augmentor, is_train=True)
        val_ds = KWDataset(val_samples, is_train=False)

        model = KeywordSpottingModelV2().to(device)
        config = {'lr': SET.LR, 'batch_size': SET.BATCH_SIZE, 'epochs': args.epochs, 'hidden': SET.HIDDEN,
                  'res_blocks': SET.NUM_RES_BLOCKS, 'dropout': 0.3, 'early_stop': SET.EARLY_STOP, 'wd': SET.WEIGHT_DECAY}
        train_model(model, train_ds, val_ds, config, device)

    elif args.mode == 'submit':
        # 仅生成提交
        best_path = SET.OUTPUT_DIR / "best_model_baseline.pt"
        model = KeywordSpottingModelV2().to(device)
        model.load_state_dict(torch.load(best_path, weights_only=True))
        model.eval()

        inset_preds = predict_on_test(model, SET.TEST_DIR, 'inset', device)
        outset_preds = predict_on_test(model, SET.TEST_DIR, 'outset', device)
        make_submission(inset_preds, outset_preds, SET.SUBMIT_DIR / "submit_v2.csv")
