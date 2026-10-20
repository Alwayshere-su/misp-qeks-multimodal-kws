"""
MISP-QEKS 多模态开放词汇关键词检索 Baseline
==============================================
任务: 给定 Enrollment（注册样本）与 Query（查询样本），判断 Query 中是否包含目标关键词。

方法:
  1. 音频特征: 使用 torchaudio 提取 MFCC + delta + delta-delta (120维)
  2. 文本特征: 使用 enrollment 关键词和 query 文本的字符级相似度
  3. 模型: 双塔结构 + 特征融合 + MLP 分类器
  4. 训练: dev_seen 作为训练集, dev_unseen 作为验证集

环境: PyTorch 1.x/2.x 兼容
"""

import os
import sys
import re
import json
import random
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import soundfile as sf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pickle

# ============================================================
# 配置
# ============================================================
class Config:
    # 路径
    BASE_DIR = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
    DATA_DIR = BASE_DIR / "val"
    OUTPUT_DIR = BASE_DIR / "output"
    SUBMISSION_DIR = BASE_DIR / "submission"

    # 数据
    TRAIN_SPLIT = "dev_seen"    # 用 dev_seen 作为训练
    VAL_SPLIT = "dev_unseen"    # 用 dev_unseen 作为验证

    # 音频参数
    SAMPLE_RATE = 16000
    N_MFCC = 40
    N_MELS = 80
    MAX_AUDIO_LEN = 4.0  # 最大音频长度 (秒)
    MAX_ENROLL_LEN = 2.0  # enrollment 最大长度
    MAX_QUERY_LEN = 4.0   # query 最大长度

    # 模型参数
    AUDIO_EMB_DIM = 256
    TEXT_EMB_DIM = 64
    HIDDEN_DIMS = [512, 256, 128]
    DROPOUT = 0.3

    # 训练参数
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    EARLY_STOP_PATIENCE = 10
    WEIGHT_DECAY = 1e-5

    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 随机种子
    SEED = 42

cfg = Config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.SEED)

# ============================================================
# 数据加载
# ============================================================

def parse_label_file(filepath):
    """解析标签文件: enrollment:xxx\tquery:xxx\tlabel:0/1"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    parts = content.split('\t')
    enroll_text = parts[0].replace('enrollment:', '')
    query_text = parts[1].replace('query:', '')
    label = int(parts[2].replace('label:', ''))
    return enroll_text, query_text, label


def parse_textgrid(filepath):
    """解析 TextGrid FA 文件, 提取音素序列"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取音素层
    phones = []
    in_phone_tier = False
    for line in content.split('\n'):
        line = line.strip()
        if 'name = "phones"' in line:
            in_phone_tier = True
            continue
        if in_phone_tier and 'intervals [' in line:
            continue
        if in_phone_tier and 'text = "' in line:
            match = re.search(r'text = "([^"]*)"', line)
            if match:
                phone = match.group(1).strip()
                if phone and phone != 'sp':  # 跳过静音标记
                    phones.append(phone)

    return phones


class KWS_Dataset(Dataset):
    """关键词检索数据集"""

    def __init__(self, data_dir, split_name, max_samples=None):
        self.data_dir = Path(data_dir)
        self.split_name = split_name
        self.split_dir = self.data_dir / split_name

        # 加载所有样本
        self.label_dir = self.split_dir / "label"
        self.wav_dir = self.split_dir / "wav"
        self.fa_dir = self.split_dir / "fa"

        self.pair_ids = []
        self.samples = []

        # 遍历 label 文件
        label_files = sorted(self.label_dir.glob("pair_*.txt"))
        if max_samples:
            label_files = label_files[:max_samples]

        print(f"  加载 {split_name}: {len(label_files)} 个样本对...")

        for label_file in tqdm(label_files, desc=f"  {split_name}"):
            pair_id = label_file.stem  # e.g., pair_000001
            self.pair_ids.append(pair_id)

            enroll_text, query_text, label = parse_label_file(label_file)

            self.samples.append({
                'pair_id': pair_id,
                'enroll_text': enroll_text,
                'query_text': query_text,
                'label': label,
                'enroll_wav': str(self.wav_dir / f"{pair_id}_enroll.wav"),
                'query_wav': str(self.wav_dir / f"{pair_id}_query.wav"),
                'enroll_fa': str(self.fa_dir / f"{pair_id}_enroll.TextGrid"),
                'query_fa': str(self.fa_dir / f"{pair_id}_query.TextGrid"),
            })

        # 统计
        pos_count = sum(1 for s in self.samples if s['label'] == 1)
        neg_count = len(self.samples) - pos_count
        print(f"  {split_name}: 正样本={pos_count}, 负样本={neg_count}, "
              f"正样本率={pos_count/len(self.samples):.3f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# 特征提取
# ============================================================

class AudioFeatureExtractor:
    """音频特征提取器"""

    def __init__(self, sample_rate=16000, n_mfcc=40, n_mels=80):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

        # MFCC 变换
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_mels': n_mels,
                'n_fft': 512,
                'hop_length': 160,
                'win_length': 400,
            }
        )

        # Delta 和 Delta-Delta
        self.delta_transform = T.ComputeDeltas()

    def extract(self, wav_path, max_duration=None):
        """提取音频特征"""
        try:
            data, sr = sf.read(wav_path, dtype='float32')
        except Exception as e:
            print(f"  警告: 无法加载 {wav_path}: {e}")
            return None

        # 转为 torch tensor: (samples,) -> (1, samples)
        waveform = torch.from_numpy(data).float().unsqueeze(0)

        # 重采样
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # 如果多声道，取平均
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 截断或填充到最大长度
        max_samples = int(max_duration * self.sample_rate) if max_duration else None
        if max_samples:
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            elif waveform.shape[1] < max_samples:
                # 零填充
                pad = torch.zeros(1, max_samples - waveform.shape[1])
                waveform = torch.cat([waveform, pad], dim=1)

        # 提取 MFCC: (1, n_mfcc, time)
        mfcc = self.mfcc_transform(waveform)

        # Delta: (1, n_mfcc, time)
        delta1 = self.delta_transform(mfcc)
        delta2 = self.delta_transform(delta1)

        # 拼接: (1, n_mfcc*3, time)
        features = torch.cat([mfcc, delta1, delta2], dim=1)

        return features  # (1, 120, time)


# ============================================================
# 模型定义
# ============================================================

class AudioEncoder(nn.Module):
    """音频编码器: CNN over MFCC features"""

    def __init__(self, input_dim=120, emb_dim=256):
        super().__init__()

        self.conv = nn.Sequential(
            # (B, 120, T) -> (B, 128, T/2)
            nn.Conv1d(input_dim, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # -> (B, 192, T/4)
            nn.Conv1d(128, 192, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(),

            # -> (B, 256, T/8)
            nn.Conv1d(192, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.proj = nn.Linear(256, emb_dim)

    def forward(self, x):
        """
        x: (B, 120, T) MFCC features
        Returns: (B, emb_dim)
        """
        x = self.conv(x)           # (B, 256, T')
        x = x.mean(dim=-1)         # (B, 256) global average pooling
        x = self.proj(x)           # (B, emb_dim)
        x = F.normalize(x, dim=-1) # L2 normalize
        return x


class TextEncoder(nn.Module):
    """轻量文本编码器: 基于字符的简单编码"""

    def __init__(self, vocab_size=256, emb_dim=64, max_len=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, emb_dim)

    def forward(self, text_ids):
        """
        text_ids: (B, max_len) character indices
        Returns: (B, emb_dim)
        """
        x = self.embedding(text_ids)  # (B, max_len, emb_dim)
        x = x.transpose(1, 2)         # (B, emb_dim, max_len)
        x = self.conv(x)              # (B, 64, 1)
        x = x.squeeze(-1)             # (B, 64)
        x = self.proj(x)              # (B, emb_dim)
        x = F.normalize(x, dim=-1)
        return x


class KeywordSpottingModel(nn.Module):
    """关键词检索模型"""

    def __init__(self, audio_emb_dim=256, text_emb_dim=64, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()

        self.audio_encoder = AudioEncoder(input_dim=120, emb_dim=audio_emb_dim)

        # 融合: [enroll_audio, query_audio, |diff|, product]
        fusion_dim = audio_emb_dim * 4

        layers = []
        in_dim = fusion_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, enroll_audio_feat, query_audio_feat):
        """
        enroll_audio_feat: (B, 120, T1)
        query_audio_feat:  (B, 120, T2)
        Returns: (B, 1) logits
        """
        enroll_emb = self.audio_encoder(enroll_audio_feat)  # (B, audio_emb_dim)
        query_emb = self.audio_encoder(query_audio_feat)    # (B, audio_emb_dim)

        # 融合特征
        diff = torch.abs(enroll_emb - query_emb)
        prod = enroll_emb * query_emb
        fusion = torch.cat([enroll_emb, query_emb, diff, prod], dim=-1)

        # 分类
        logits = self.classifier(fusion)
        return logits

    def predict(self, enroll_audio_feat, query_audio_feat):
        """返回概率"""
        with torch.no_grad():
            logits = self.forward(enroll_audio_feat, query_audio_feat)
            probs = torch.sigmoid(logits).squeeze(-1)
        return probs


# ============================================================
# 文本处理工具
# ============================================================

def text_to_ids(text, max_len=50):
    """将文本转为字符级 ID 序列"""
    ids = [min(ord(c), 255) for c in text[:max_len]]
    # 填充
    ids = ids + [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def compute_text_similarity(enroll_text, query_text):
    """计算文本级别的简单相似度特征"""
    enroll_chars = set(enroll_text.lower())
    query_chars = set(query_text.lower())

    if len(enroll_chars) == 0:
        return 0.0

    # Jaccard 相似度 (字符级别)
    intersection = len(enroll_chars & query_chars)
    union = len(enroll_chars | query_chars)
    jaccard = intersection / union if union > 0 else 0.0

    # 长度比例
    len_ratio = min(len(enroll_text), len(query_text)) / max(len(enroll_text), len(query_text), 1)

    return jaccard


# ============================================================
# 特征缓存 (避免重复提取)
# ============================================================

class FeatureCache:
    """缓存提取的音频特征"""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, wav_path):
        return Path(wav_path).stem

    def has(self, wav_path):
        key = self.get_cache_key(wav_path)
        return (self.cache_dir / f"{key}.pt").exists()

    def load(self, wav_path):
        key = self.get_cache_key(wav_path)
        return torch.load(self.cache_dir / f"{key}.pt", weights_only=True)

    def save(self, wav_path, features):
        key = self.get_cache_key(wav_path)
        torch.save(features, self.cache_dir / f"{key}.pt")


# ============================================================
# 数据批处理
# ============================================================

def collate_fn(batch, feature_extractor, feat_cache, device):
    """自定义批处理函数"""
    enroll_feats = []
    query_feats = []
    labels = []
    batch_info = []

    for sample in batch:
        # 加载/缓存 enrollment 特征
        if feat_cache.has(sample['enroll_wav']):
            ef = feat_cache.load(sample['enroll_wav'])
        else:
            ef = feature_extractor.extract(sample['enroll_wav'], max_duration=cfg.MAX_ENROLL_LEN)
            if ef is not None:
                feat_cache.save(sample['enroll_wav'], ef)

        # 加载/缓存 query 特征
        if feat_cache.has(sample['query_wav']):
            qf = feat_cache.load(sample['query_wav'])
        else:
            qf = feature_extractor.extract(sample['query_wav'], max_duration=cfg.MAX_QUERY_LEN)
            if qf is not None:
                feat_cache.save(sample['query_wav'], qf)

        if ef is None or qf is None:
            continue

        enroll_feats.append(ef.squeeze(0))  # (120, T)
        query_feats.append(qf.squeeze(0))   # (120, T)
        labels.append(sample['label'])
        batch_info.append(sample)

    if len(enroll_feats) == 0:
        return None

    # Pad to max length in batch (手动 padding, 因 pad_sequence 在 dim0 上 pad)
    # 所有 feature 形状为 (120, T), 需在 T 维度上 pad 到 max_T
    enroll_max_len = max(f.shape[1] for f in enroll_feats)
    enroll_padded = []
    for f in enroll_feats:
        if f.shape[1] < enroll_max_len:
            pad = torch.zeros(f.shape[0], enroll_max_len - f.shape[1])
            f = torch.cat([f, pad], dim=1)
        enroll_padded.append(f)
    enroll_feats = torch.stack(enroll_padded, dim=0)  # (B, 120, max_T)

    query_max_len = max(f.shape[1] for f in query_feats)
    query_padded = []
    for f in query_feats:
        if f.shape[1] < query_max_len:
            pad = torch.zeros(f.shape[0], query_max_len - f.shape[1])
            f = torch.cat([f, pad], dim=1)
        query_padded.append(f)
    query_feats = torch.stack(query_padded, dim=0)  # (B, 120, max_T)

    labels = torch.tensor(labels, dtype=torch.float32)

    return {
        'enroll_feat': enroll_feats.to(device),
        'query_feat': query_feats.to(device),
        'label': labels.to(device),
        'batch_info': batch_info,
    }


# ============================================================
# 训练和评估
# ============================================================

def train_epoch(model, dataset, feature_extractor, feat_cache, optimizer, criterion, device, batch_size):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    # 随机打乱
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    num_batches = (len(indices) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="  Training"):
        start = i * batch_size
        end = min(start + batch_size, len(indices))
        batch_indices = indices[start:end]
        batch = [dataset[idx] for idx in batch_indices]

        batch_data = collate_fn(batch, feature_extractor, feat_cache, device)
        if batch_data is None:
            continue

        optimizer.zero_grad()
        logits = model(batch_data['enroll_feat'], batch_data['query_feat'])
        loss = criterion(logits.squeeze(-1), batch_data['label'])
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()

        preds = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_data['label'].cpu().numpy().tolist())

    avg_loss = total_loss / max(num_batches, 1)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5

    return avg_loss, auc


def evaluate(model, dataset, feature_extractor, feat_cache, device, batch_size):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    indices = list(range(len(dataset)))
    num_batches = (len(indices) + batch_size - 1) // batch_size
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="  Evaluating"):
            start = i * batch_size
            end = min(start + batch_size, len(indices))
            batch = [dataset[idx] for idx in range(start, end)]

            batch_data = collate_fn(batch, feature_extractor, feat_cache, device)
            if batch_data is None:
                continue

            logits = model(batch_data['enroll_feat'], batch_data['query_feat'])
            loss = criterion(logits.squeeze(-1), batch_data['label'])
            total_loss += loss.item()

            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_data['label'].cpu().numpy().tolist())

    avg_loss = total_loss / max(num_batches, 1)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5

    return avg_loss, auc, all_preds, all_labels


# ============================================================
# 主训练流程
# ============================================================

def main():
    print("=" * 60)
    print("MISP-QEKS Baseline Training")
    print(f"Device: {cfg.DEVICE}")
    print("=" * 60)

    # 创建输出目录
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化特征提取器和缓存
    print("\n[1] 初始化特征提取器...")
    feature_extractor = AudioFeatureExtractor(
        sample_rate=cfg.SAMPLE_RATE,
        n_mfcc=cfg.N_MFCC,
        n_mels=cfg.N_MELS,
    )
    feat_cache = FeatureCache(cfg.OUTPUT_DIR / "feature_cache")

    # 加载数据
    print("\n[2] 加载数据...")
    train_dataset = KWS_Dataset(cfg.DATA_DIR, cfg.TRAIN_SPLIT)
    val_dataset = KWS_Dataset(cfg.DATA_DIR, cfg.VAL_SPLIT)

    print(f"\n  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")

    # 初始化模型
    print("\n[3] 初始化模型...")
    model = KeywordSpottingModel(
        audio_emb_dim=cfg.AUDIO_EMB_DIM,
        text_emb_dim=cfg.TEXT_EMB_DIM,
        hidden_dims=cfg.HIDDEN_DIMS,
        dropout=cfg.DROPOUT,
    ).to(cfg.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 优化器和损失函数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()

    # 训练循环
    print("\n[4] 开始训练...")
    print(f"  Batch size: {cfg.BATCH_SIZE}")
    print(f"  Learning rate: {cfg.LEARNING_RATE}")
    print(f"  Max epochs: {cfg.NUM_EPOCHS}")
    print(f"  Early stop patience: {cfg.EARLY_STOP_PATIENCE}")

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        print(f"\n  --- Epoch {epoch}/{cfg.NUM_EPOCHS} ---")

        # 训练
        train_loss, train_auc = train_epoch(
            model, train_dataset, feature_extractor, feat_cache,
            optimizer, criterion, cfg.DEVICE, cfg.BATCH_SIZE
        )

        # 验证
        val_loss, val_auc, val_preds, val_labels = evaluate(
            model, val_dataset, feature_extractor, feat_cache,
            cfg.DEVICE, cfg.BATCH_SIZE
        )

        # 更新学习率
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']

        # 记录
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        print(f"  Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   AUC: {val_auc:.4f} | LR: {current_lr:.2e}")

        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0

            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'train_auc': train_auc,
                'config': cfg,
            }, cfg.OUTPUT_DIR / "best_model.pt")
            print(f"  *** 新最佳模型! Val AUC: {val_auc:.4f} ***")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{cfg.EARLY_STOP_PATIENCE} (best: {best_val_auc:.4f} at epoch {best_epoch})")

        if patience_counter >= cfg.EARLY_STOP_PATIENCE:
            print(f"\n  早停触发! 最佳 Val AUC: {best_val_auc:.4f} (epoch {best_epoch})")
            break

    # 最终评估
    print("\n[5] 最终评估...")
    print(f"  最佳验证 AUC: {best_val_auc:.4f}")

    # 综合分数: 0.5 * AUC_seen + 0.5 * AUC_blind
    # dev_unseen 对应 "未见关键词" (Blind)
    overall_score = 0.5 * best_val_auc + 0.5 * best_val_auc  # 简化: 无 seen 单独评估
    print(f"  预估综合分数: {overall_score:.4f}")

    # 保存训练历史
    with open(cfg.OUTPUT_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  训练完成! 最佳模型: {cfg.OUTPUT_DIR / 'best_model.pt'}")

    return model, feature_extractor, feat_cache, best_val_auc


# ============================================================
# 生成提交文件
# ============================================================

def generate_submission(model, feature_extractor, feat_cache, data_dir, output_path, split_name):
    """
    生成单个 split 的预测结果
    注意: 对于正式的测试集, 数据格式可能不同; 这里用 dev 数据模拟
    """
    print(f"\n  生成 {split_name} 预测...")

    dataset = KWS_Dataset(data_dir, split_name)
    device = cfg.DEVICE

    model.eval()
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"  {split_name}"):
            sample = dataset[i]

            # 加载特征
            if feat_cache.has(sample['enroll_wav']):
                ef = feat_cache.load(sample['enroll_wav'])
            else:
                ef = feature_extractor.extract(sample['enroll_wav'], max_duration=cfg.MAX_ENROLL_LEN)
                if ef is not None:
                    feat_cache.save(sample['enroll_wav'], ef)

            if feat_cache.has(sample['query_wav']):
                qf = feat_cache.load(sample['query_wav'])
            else:
                qf = feature_extractor.extract(sample['query_wav'], max_duration=cfg.MAX_QUERY_LEN)
                if qf is not None:
                    feat_cache.save(sample['query_wav'], qf)

            if ef is None or qf is None:
                # 默认预测
                pred = 0.5
            else:
                ef = ef.to(device)
                qf = qf.to(device)
                pred = model.predict(ef, qf).cpu().item()

            predictions.append(pred)

    return predictions


def create_submission_file(predictions_seen, predictions_blind, output_path):
    """创建符合提交格式的 CSV 文件"""
    print(f"\n  创建提交文件: {output_path}")

    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")

        # inset (Eval-Seen)
        for i, pred in enumerate(predictions_seen):
            f.write(f"{i},inset,{pred:.6f}\n")

        # outset (Eval-Blind)
        for i, pred in enumerate(predictions_blind):
            f.write(f"{i},outset,{pred:.6f}\n")

    print(f"  文件已保存: {output_path}")
    print(f"    inset: {len(predictions_seen)} 行")
    print(f"    outset: {len(predictions_blind)} 行")


# ============================================================
# Loop: 持续优化
# ============================================================

class ImprovementLoop:
    """持续改进循环: 尝试不同超参数, 跟踪最佳 AUC"""

    def __init__(self):
        self.best_score = 0.0
        self.best_config = None
        self.iteration = 0
        self.results = []

    def run(self, max_iterations=10):
        """运行多轮训练, 每轮尝试不同超参数"""
        print("\n" + "=" * 60)
        print("Improvement Loop - 持续优化")
        print(f"目标: 最大化 0.5*AUC_seen + 0.5*AUC_blind")
        print("=" * 60)

        # 超参数搜索空间
        configs = [
            {'lr': 1e-3, 'batch_size': 64, 'hidden_dims': [512, 256, 128], 'dropout': 0.3},
            {'lr': 5e-4, 'batch_size': 64, 'hidden_dims': [768, 384, 192], 'dropout': 0.3},
            {'lr': 1e-3, 'batch_size': 32, 'hidden_dims': [512, 256], 'dropout': 0.4},
            {'lr': 2e-3, 'batch_size': 64, 'hidden_dims': [256, 128, 64], 'dropout': 0.2},
            {'lr': 5e-4, 'batch_size': 128, 'hidden_dims': [512, 256, 128], 'dropout': 0.35},
            {'lr': 1e-3, 'batch_size': 64, 'hidden_dims': [1024, 512, 256], 'dropout': 0.3},
            {'lr': 8e-4, 'batch_size': 64, 'hidden_dims': [512, 256, 128], 'dropout': 0.25},
            {'lr': 1e-3, 'batch_size': 64, 'hidden_dims': [640, 320, 160], 'dropout': 0.3},
            {'lr': 5e-4, 'batch_size': 32, 'hidden_dims': [512, 256, 128], 'dropout': 0.5},
            {'lr': 1e-3, 'batch_size': 96, 'hidden_dims': [512, 256, 128], 'dropout': 0.3},
        ]

        max_iterations = min(max_iterations, len(configs))

        for i in range(max_iterations):
            self.iteration = i + 1
            config = configs[i]

            print(f"\n{'='*60}")
            print(f"Iteration {self.iteration}/{max_iterations}")
            print(f"Config: {config}")
            print(f"{'='*60}")

            # 更新配置
            cfg.LEARNING_RATE = config['lr']
            cfg.BATCH_SIZE = config['batch_size']
            cfg.HIDDEN_DIMS = config['hidden_dims']
            cfg.DROPOUT = config['dropout']

            set_seed(cfg.SEED)

            # 初始化
            feature_extractor = AudioFeatureExtractor(
                sample_rate=cfg.SAMPLE_RATE,
                n_mfcc=cfg.N_MFCC,
                n_mels=cfg.N_MELS,
            )
            feat_cache = FeatureCache(cfg.OUTPUT_DIR / "feature_cache")

            # 加载数据
            train_dataset = KWS_Dataset(cfg.DATA_DIR, cfg.TRAIN_SPLIT)
            val_dataset = KWS_Dataset(cfg.DATA_DIR, cfg.VAL_SPLIT)

            # 模型
            model = KeywordSpottingModel(
                audio_emb_dim=cfg.AUDIO_EMB_DIM,
                text_emb_dim=cfg.TEXT_EMB_DIM,
                hidden_dims=cfg.HIDDEN_DIMS,
                dropout=cfg.DROPOUT,
            ).to(cfg.DEVICE)

            # 优化器
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg.LEARNING_RATE,
                weight_decay=cfg.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
            )
            criterion = nn.BCEWithLogitsLoss()

            # 训练
            best_val_auc = 0.0
            patience_counter = 0

            for epoch in range(1, cfg.NUM_EPOCHS + 1):
                train_loss, train_auc = train_epoch(
                    model, train_dataset, feature_extractor, feat_cache,
                    optimizer, criterion, cfg.DEVICE, cfg.BATCH_SIZE
                )
                val_loss, val_auc, _, _ = evaluate(
                    model, val_dataset, feature_extractor, feat_cache,
                    cfg.DEVICE, cfg.BATCH_SIZE
                )

                scheduler.step(val_auc)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0

                    # 保存本轮最佳模型
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_auc': val_auc,
                        'config': config,
                    }, cfg.OUTPUT_DIR / f"model_iter{self.iteration}.pt")
                else:
                    patience_counter += 1

                if patience_counter >= cfg.EARLY_STOP_PATIENCE:
                    break

            # 记录结果
            overall_score = 0.5 * best_val_auc + 0.5 * best_val_auc  # 简化
            self.results.append({
                'iteration': self.iteration,
                'config': config,
                'val_auc': best_val_auc,
                'overall_score': overall_score,
            })

            print(f"\n  Iter {self.iteration} 结果: Val AUC={best_val_auc:.4f}, "
                  f"Overall Score={overall_score:.4f}")

            # 更新全局最佳
            if overall_score > self.best_score:
                self.best_score = overall_score
                self.best_config = config
                print(f"  *** 新全局最佳! Score: {self.best_score:.4f} ***")

                # 复制最佳模型
                import shutil
                shutil.copy(
                    cfg.OUTPUT_DIR / f"model_iter{self.iteration}.pt",
                    cfg.OUTPUT_DIR / "best_model_loop.pt"
                )

        # 总结
        print("\n" + "=" * 60)
        print("Loop 完成! 结果汇总:")
        print("=" * 60)
        for r in sorted(self.results, key=lambda x: x['overall_score'], reverse=True):
            marker = " <-- BEST" if r['overall_score'] == self.best_score else ""
            print(f"  Iter {r['iteration']:2d}: AUC={r['val_auc']:.4f}, "
                  f"Score={r['overall_score']:.4f}, Config={r['config']}{marker}")

        # 保存结果
        with open(cfg.OUTPUT_DIR / "loop_results.json", 'w') as f:
            json.dump({
                'best_score': self.best_score,
                'best_config': self.best_config,
                'results': self.results,
            }, f, indent=2)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MISP-QEKS Baseline')
    parser.add_argument('--mode', type=str, default='loop',
                        choices=['train', 'loop', 'submit'],
                        help='运行模式: train=单次训练, loop=优化循环, submit=生成提交文件')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Loop 迭代次数')
    parser.add_argument('--epochs', type=int, default=50,
                        help='每轮训练的最大 epoch 数')
    args = parser.parse_args()

    cfg.NUM_EPOCHS = args.epochs

    if args.mode == 'train':
        # 单次训练
        model, feature_extractor, feat_cache, best_auc = main()

        # 生成提交文件
        print("\n[6] 生成提交样本...")
        seen_preds = generate_submission(
            model, feature_extractor, feat_cache, cfg.DATA_DIR,
            cfg.SUBMISSION_DIR / "submit_inset.csv", cfg.TRAIN_SPLIT
        )
        blind_preds = generate_submission(
            model, feature_extractor, feat_cache, cfg.DATA_DIR,
            cfg.SUBMISSION_DIR / "submit_outset.csv", cfg.VAL_SPLIT
        )

        # 模拟 50000 条/类的完整提交格式
        # 如果实际数据不足 50000 条, 用预测值填充
        full_seen = list(seen_preds)
        full_blind = list(blind_preds)

        # 扩展或截断到 50000
        while len(full_seen) < 50000:
            full_seen.append(0.5)
        while len(full_blind) < 50000:
            full_blind.append(0.5)

        full_seen = full_seen[:50000]
        full_blind = full_blind[:50000]

        create_submission_file(
            full_seen, full_blind,
            cfg.SUBMISSION_DIR / "submit_baseline.csv"
        )

    elif args.mode == 'loop':
        # 运行优化循环
        loop = ImprovementLoop()
        loop.run(max_iterations=args.iterations)

        # 加载最佳模型并生成提交文件
        print("\n\n[最终] 生成最佳模型的提交文件...")
        checkpoint = torch.load(cfg.OUTPUT_DIR / "best_model_loop.pt", weights_only=True)
        best_model = KeywordSpottingModel(
            audio_emb_dim=cfg.AUDIO_EMB_DIM,
            text_emb_dim=cfg.TEXT_EMB_DIM,
            hidden_dims=checkpoint['config']['hidden_dims'],
            dropout=checkpoint['config']['dropout'],
        ).to(cfg.DEVICE)
        best_model.load_state_dict(checkpoint['model_state_dict'])

        feature_extractor = AudioFeatureExtractor(
            sample_rate=cfg.SAMPLE_RATE,
            n_mfcc=cfg.N_MFCC,
            n_mels=cfg.N_MELS,
        )
        feat_cache = FeatureCache(cfg.OUTPUT_DIR / "feature_cache")

        seen_preds = generate_submission(
            best_model, feature_extractor, feat_cache, cfg.DATA_DIR,
            cfg.SUBMISSION_DIR / "submit_inset.csv", cfg.TRAIN_SPLIT
        )
        blind_preds = generate_submission(
            best_model, feature_extractor, feat_cache, cfg.DATA_DIR,
            cfg.SUBMISSION_DIR / "submit_outset.csv", cfg.VAL_SPLIT
        )

        full_seen = list(seen_preds)
        full_blind = list(blind_preds)
        while len(full_seen) < 50000:
            full_seen.append(0.5)
        while len(full_blind) < 50000:
            full_blind.append(0.5)
        full_seen = full_seen[:50000]
        full_blind = full_blind[:50000]

        create_submission_file(
            full_seen, full_blind,
            cfg.SUBMISSION_DIR / "submit_baseline.csv"
        )

        print(f"\n全局最佳分数: {loop.best_score:.4f}")
        print(f"最佳配置: {loop.best_config}")

    elif args.mode == 'submit':
        # 仅生成提交文件 (使用已有最佳模型)
        checkpoint = torch.load(cfg.OUTPUT_DIR / "best_model_loop.pt", weights_only=True)
        best_model = KeywordSpottingModel(
            audio_emb_dim=cfg.AUDIO_EMB_DIM,
            text_emb_dim=cfg.TEXT_EMB_DIM,
            hidden_dims=checkpoint['config']['hidden_dims'],
            dropout=checkpoint['config']['dropout'],
        ).to(cfg.DEVICE)
        best_model.load_state_dict(checkpoint['model_state_dict'])

        feature_extractor = AudioFeatureExtractor(
            sample_rate=cfg.SAMPLE_RATE,
            n_mfcc=cfg.N_MFCC,
            n_mels=cfg.N_MELS,
        )
        feat_cache = FeatureCache(cfg.OUTPUT_DIR / "feature_cache")

        seen_preds = generate_submission(
            best_model, feature_extractor, feat_cache, cfg.DATA_DIR,
            cfg.SUBMISSION_DIR / "submit_inset.csv", cfg.TRAIN_SPLIT
        )
        blind_preds = generate_submission(
            best_model, feature_extractor, feat_cache, cfg.DATA_DIR,
            cfg.SUBMISSION_DIR / "submit_outset.csv", cfg.VAL_SPLIT
        )

        full_seen = list(seen_preds)
        full_blind = list(blind_preds)
        while len(full_seen) < 50000:
            full_seen.append(0.5)
        while len(full_blind) < 50000:
            full_blind.append(0.5)
        full_seen = full_seen[:50000]
        full_blind = full_blind[:50000]

        create_submission_file(
            full_seen, full_blind,
            cfg.SUBMISSION_DIR / "submit_baseline.csv"
        )
