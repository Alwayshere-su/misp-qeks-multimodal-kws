"""
MISP-QEKS Baseline V4 — Loop 1: 文本模态 + Focal Loss
=====================================================
基于 V3，新增:
1. 字符级 TextEncoder — 编码 enrollment 关键词文本 (如 "cathedral")
2. 多模态融合 — audio_emb + text_emb → fused enrollment
3. Focal Loss — 关注难分样本
4. 字符词表 — 从所有 enrollment 单词自动构建
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
OUT = BASE / "output_v4"
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
# 字符词表 — 从所有 enrollment 文本构建
# ============================================================

def build_vocab(corpus_words):
    """从英文单词列表构建字符级词表"""
    chars = set()
    for w in corpus_words:
        for c in w.lower():
            chars.add(c)
    # 排序: 先特殊 token, 再数字, 再字母
    vocab = ['<pad>', '<unk>'] + sorted(ch for ch in chars if not ch.isalpha())
    vocab += sorted(ch for ch in chars if ch.isalpha())
    char2idx = {c: i for i, c in enumerate(vocab)}
    return vocab, char2idx

def tokenize(word, char2idx, max_len=20):
    """将单词转为字符ID序列"""
    ids = [char2idx.get(c, char2idx['<unk>']) for c in word.lower()]
    # 截断或 pad
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [char2idx['<pad>']] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

# ============================================================
# 文本编码器
# ============================================================

class TextEncoder(nn.Module):
    """字符级文本编码器 — 编码 enrollment 关键词"""
    def __init__(self, vocab_size, emb_dim=64, hidden=256):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
        )
        self.attn_pool = nn.Conv1d(256, 1, 1)
        self.proj = nn.Linear(256, hidden)

    def forward(self, tokens):
        """tokens: (B, max_len)"""
        x = self.char_emb(tokens).transpose(1, 2)  # (B, emb_dim, L)
        x = self.conv(x)                            # (B, 256, L)
        w = self.attn_pool(x).softmax(-1)           # (B, 1, L)
        x = (w * x).sum(-1)                         # (B, 256)
        return self.proj(x)                         # (B, hidden)

# ============================================================
# 音频编码器 (沿用 V3)
# ============================================================

class AudioEncoderV3(nn.Module):
    """高效音频编码器"""
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
        x = self.encoder(x)
        w = self.attn(x).softmax(-1)
        global_e = (w * x).sum(-1)
        global_e = self.proj(global_e)
        frames = x.transpose(1, 2)
        frames = self.proj(frames)
        return global_e, frames

# ============================================================
# V4 多模态关键词检索器
# ============================================================

class KWSpotterV4(nn.Module):
    """V4 关键词检索器 — 音频 + 文本双模态"""
    def __init__(self, hidden=256, vocab_size=64, text_dim=256):
        super().__init__()
        self.audio_encoder = AudioEncoderV3(120, hidden, hidden)
        self.text_encoder = TextEncoder(vocab_size, emb_dim=64, hidden=text_dim)

        # 融合: audio_emb (hidden) + text_emb (text_dim) -> hidden
        self.fusion = nn.Sequential(
            nn.Linear(hidden + text_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
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

    def forward(self, ef, qf, enroll_tokens):
        """
        ef: (B, 120, Te) — enrollment audio
        qf: (B, 120, Tq) — query audio
        enroll_tokens: (B, max_len) — enrollment keyword text tokens
        """
        # Audio encoding
        e_audio, e_frames = self.audio_encoder(ef)  # (B, H), (B, Te', H)
        q_global, q_frames = self.audio_encoder(qf)  # (B, H), (B, Tq', H)

        # Text encoding
        e_text = self.text_encoder(enroll_tokens)  # (B, text_dim)

        # Fusion: audio + text -> enrollment embedding
        e_global = self.fusion(torch.cat([e_audio, e_text], dim=-1))  # (B, H)

        # Cross-attention: fused enrollment attends to query frames
        q_attn, _ = self.match_attn(
            e_global.unsqueeze(1), q_frames, q_frames
        )  # (B, 1, H)
        q_attn = self.match_norm(q_attn.squeeze(1))

        # Feature fusion for classification
        diff = torch.abs(e_global - q_attn)
        prod = e_global * q_attn
        fused = torch.cat([e_global, q_attn, diff, prod], dim=-1)

        logit = self.cls(fused)
        return logit

    def predict(self, ef, qf, enroll_tokens):
        with torch.no_grad():
            return torch.sigmoid(self.forward(ef, qf, enroll_tokens)).squeeze(-1)

# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss — 自动关注难分样本"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets, reduction='none')
        pt = torch.exp(-bce)
        # Alpha-weight: higher weight for positive class
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()

# ============================================================
# MFCC 特征提取
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
    return feat.squeeze(0)

# ============================================================
# 数据预提取 — 新增文本提取
# ============================================================

def parse_enroll_text_from_label(label_path):
    """从 label 文件提取 enrollment 关键词文本"""
    with open(label_path) as f:
        content = f.read().strip()
    # 格式: enrollment:<word>\tquery:<word>\tlabel:0/1
    m = re.search(r'enrollment:(\S+)', content)
    if m:
        return m.group(1)
    return '<unk>'

def pre_extract_all(split_dir, max_enroll=2.0, max_query=4.0):
    """预提取特征 + 文本"""
    d = Path(split_dir)
    labels_path = d / "label"
    wav_dir = d / "wav"
    data = {}
    files = sorted(labels_path.glob("pair_*.txt"))
    print(f"  预提取 {d.name}: {len(files)} pairs...")

    for lf in tqdm(files, desc=f"  {d.name}"):
        pid = lf.stem
        with open(lf) as f:
            parts = f.read().strip().split('\t')
        label = int(parts[2].replace('label:', ''))

        ef = extract_mfcc_feats(wav_dir / f"{pid}_enroll.wav", max_enroll)
        qf = extract_mfcc_feats(wav_dir / f"{pid}_query.wav", max_query)
        enroll_word = parse_enroll_text_from_label(lf)

        data[pid] = {
            'enroll_feat': ef,
            'query_feat': qf,
            'label': label,
            'enroll_text': enroll_word,  # 新增
        }
    return data

def pre_extract_test_with_text(test_dir, split_type, enroll_text_csv, max_enroll=2.0, max_query=4.0):
    """提取测试集特征 + 文本"""
    d = Path(test_dir)
    if split_type == 'inset':
        wav_dir = d / "eval_seen_unlabeled" / "wav"
    else:
        wav_dir = d / "eval_unseen_unlabeled" / "wav"

    # 加载 enrollment text
    text_map = {}
    with open(enroll_text_csv, encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            pid, txt = line.strip().split(',', 1)
            text_map[pid] = txt

    files = sorted(wav_dir.glob("pair_*_enroll.wav"))
    data = {}
    print(f"  预提取 test/{split_type}: {len(files)} pairs...")

    for f in tqdm(files, desc=f"  test/{split_type}"):
        pid = f.stem.replace('_enroll', '')
        ef = extract_mfcc_feats(wav_dir / f"{pid}_enroll.wav", max_enroll)
        qf = extract_mfcc_feats(wav_dir / f"{pid}_query.wav", max_query)
        data[pid] = {
            'enroll_feat': ef,
            'query_feat': qf,
            'enroll_text': text_map.get(pid, '<unk>'),  # 新增
        }
    return data

# ============================================================
# 数据集 (支持文本)
# ============================================================

class MemKWDatasetV4(Dataset):
    """内存数据集 V4 — 支持文本 token"""
    def __init__(self, data_dict, char2idx, max_text_len=20, augment=False, noise_feats=None):
        self.items = list(data_dict.items())
        self.char2idx = char2idx
        self.max_text_len = max_text_len
        self.augment = augment
        self.noise_feats = noise_feats or []

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        pid, d = self.items[idx]
        ef = d['enroll_feat']
        qf = d['query_feat']
        label = float(d.get('label', 0.5))
        enroll_word = d.get('enroll_text', '<unk>')

        # Tokenize text
        tokens = tokenize(enroll_word, self.char2idx, self.max_text_len)

        # 噪声增强
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

        return {
            'enroll_feat': ef, 'query_feat': qf, 'label': label,
            'enroll_tokens': tokens, 'pid': pid,
        }

def pad_collate_fn_v4(batch):
    """V4 collate — 包含文本 tokens"""
    efs, qfs, tokens_list, labels, pids = [], [], [], [], []
    for b in batch:
        efs.append(b['enroll_feat'].transpose(0, 1))
        qfs.append(b['query_feat'].transpose(0, 1))
        tokens_list.append(b['enroll_tokens'])
        labels.append(b['label'])
        pids.append(b['pid'])
    efs = nn.utils.rnn.pad_sequence(efs, batch_first=True).transpose(1, 2)
    qfs = nn.utils.rnn.pad_sequence(qfs, batch_first=True).transpose(1, 2)
    tokens_batch = torch.stack(tokens_list)
    return {
        'ef': efs, 'qf': qfs, 'enroll_tokens': tokens_batch,
        'label': torch.tensor(labels, dtype=torch.float32),
        'pid': pids
    }

def balance_sampler(dataset):
    pos_idx = [i for i, (_, d) in enumerate(dataset.items) if d.get('label', 0) == 1]
    neg_idx = [i for i, (_, d) in enumerate(dataset.items) if d.get('label', 0) == 0]
    if not pos_idx: return None
    w = max(1, len(neg_idx) // len(pos_idx))
    indices = neg_idx + pos_idx * w
    random.shuffle(indices)
    return indices

# ============================================================
# 训练 (Focal Loss)
# ============================================================

def train_one_epoch_v4(model, ds, optimizer, bs, device, criterion, contrastive_weight=0.1):
    model.train()
    indices = balance_sampler(ds) or list(range(len(ds)))
    random.shuffle(indices)

    total_loss, n = 0, 0
    preds, labels = [], []

    for i in tqdm(range(0, len(indices), bs), desc="  Train", leave=False):
        batch_idx = indices[i:i+bs]
        batch = pad_collate_fn_v4([ds[idx] for idx in batch_idx])

        ef = batch['ef'].to(device)
        qf = batch['qf'].to(device)
        et = batch['enroll_tokens'].to(device)
        y = batch['label'].to(device)

        logit = model(ef, qf, et)
        loss = criterion(logit.squeeze(-1), y)

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
def evaluate_v4(model, ds, bs, device):
    model.eval()
    preds, labels = [], []
    for i in tqdm(range(0, len(ds), bs), desc="  Eval", leave=False):
        batch = pad_collate_fn_v4([ds[idx] for idx in range(i, min(i+bs, len(ds)))])
        ef = batch['ef'].to(device)
        qf = batch['qf'].to(device)
        et = batch['enroll_tokens'].to(device)
        prob = model.predict(ef, qf, et)
        preds.extend(prob.cpu().tolist())
        labels.extend(batch['label'].tolist())
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return auc, preds

def train_full_v4(model, train_ds, val_ds, cfg_hp, device):
    optimizer = optim.AdamW(model.parameters(), lr=cfg_hp['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=8, factor=0.5)
    criterion = FocalLoss(alpha=cfg_hp.get('focal_alpha', 0.25),
                          gamma=cfg_hp.get('focal_gamma', 2.0))

    best_auc, best_ep, patience = 0.5, 0, 0

    for ep in range(1, cfg_hp['epochs'] + 1):
        t_loss, t_auc = train_one_epoch_v4(model, train_ds, optimizer, cfg_hp['bs'], device, criterion)
        v_auc, _ = evaluate_v4(model, val_ds, cfg_hp['bs'], device)
        scheduler.step(v_auc)

        print(f"  Ep {ep:3d}: loss={t_loss:.4f} | train_AUC={t_auc:.4f} | val_AUC={v_auc:.4f}")

        if v_auc > best_auc:
            best_auc = v_auc; best_ep = ep; patience = 0
            torch.save(model.state_dict(), OUT / f"best_{cfg_hp['tag']}.pt")
        else:
            patience += 1

        if patience >= cfg_hp.get('patience', 15):
            print(f"  早停 @ ep {ep}")
            break

    return best_auc, best_ep

# ============================================================
# 预测测试集
# ============================================================

@torch.no_grad()
def predict_test_v4(model, test_data, char2idx, device):
    model.eval()
    preds = {}
    max_len = 20
    for pid, d in tqdm(test_data.items(), desc="  Predict"):
        ef = d['enroll_feat'].unsqueeze(0).to(device)
        qf = d['query_feat'].unsqueeze(0).to(device)
        word = d.get('enroll_text', '<unk>')
        tokens = tokenize(word, char2idx, max_len).unsqueeze(0).to(device)
        prob = model.predict(ef, qf, tokens).cpu().item()
        preds[pid] = prob
    return preds

def make_csv_v4(inset_preds, outset_preds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},inset,{inset_preds.get(pid, 0.5):.6f}\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},outset,{outset_preds.get(pid, 0.5):.6f}\n")
    print(f"  -> {path}")

# ============================================================
# 主 Loop
# ============================================================

def run_loop():
    print(f"Device: {DEVICE}")

    # ---- 1. 预提取特征 ----
    print("\n[Phase 1] 预提取特征 + 文本...")
    cache_path = OUT / "all_features_v4.pkl"

    if cache_path.exists():
        print(f"  加载缓存: {cache_path}")
        with open(cache_path, 'rb') as f:
            all_data = pickle.load(f)
    else:
        all_data = {
            'train': pre_extract_all(DATA / "dev_seen"),
            'val': pre_extract_all(DATA / "dev_unseen"),
            'test_inset': None,
            'test_outset': None,
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(all_data, f)
        print(f"  特征已缓存: {cache_path}")

    # ---- 1.5 构建字符词表 ----
    print("\n[Phase 1.5] 构建字符词表...")
    all_words = set()
    for split in ['train', 'val']:
        for pid, d in all_data[split].items():
            all_words.add(d.get('enroll_text', '<unk>'))
    # 也从测试 CSV 收集词汇
    for csv_path in [TEST / "eval_seen_enroll_text.csv", TEST / "eval_unseen_enroll_text.csv"]:
        if csv_path.exists():
            with open(csv_path, encoding='utf-8') as f:
                next(f)
                for line in f:
                    _, txt = line.strip().split(',', 1)
                    all_words.add(txt)
    vocab, char2idx = build_vocab(all_words)
    print(f"  词表大小: {len(vocab)} (来自 {len(all_words)} 个唯一单词)")

    # ---- 1.6 噪声特征 ----
    noise_feats = []
    noise_cache = AUX / "noise"
    noise_pkl = OUT / "noise_feats_v4.pkl"
    if noise_pkl.exists():
        with open(noise_pkl, 'rb') as f:
            noise_feats = pickle.load(f)
    else:
        try:
            for nd in noise_cache.iterdir():
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
    train_ds = MemKWDatasetV4(all_data['train'], char2idx, augment=True, noise_feats=noise_feats)
    val_ds = MemKWDatasetV4(all_data['val'], char2idx, augment=False)

    pos = sum(1 for _, d in train_ds.items if d.get('label', 0) == 1)
    print(f"  Train: {len(train_ds)} ({pos} pos)")
    print(f"  Val:   {len(val_ds)}")

    # ---- 3. 超参数搜索 ----
    print("\n[Phase 3] 超参数搜索 Loop...")

    hp_grid = [
        # 基础配置
        {'lr': 1e-3, 'bs': 64, 'epochs': 40, 'hidden': 320, 'text_dim': 256, 'focal_alpha': 0.25, 'focal_gamma': 2.0, 'patience': 12, 'tag': 'v4_base'},
        # 宽配置
        {'lr': 5e-4, 'bs': 64, 'epochs': 50, 'hidden': 384, 'text_dim': 320, 'focal_alpha': 0.25, 'focal_gamma': 1.5, 'patience': 15, 'tag': 'v4_wide'},
        # 大文本编码器
        {'lr': 3e-4, 'bs': 48, 'epochs': 50, 'hidden': 320, 'text_dim': 384, 'focal_alpha': 0.30, 'focal_gamma': 2.0, 'patience': 12, 'tag': 'v4_bigtext'},
        # 高 focal gamma — 关注难样本
        {'lr': 5e-4, 'bs': 64, 'epochs': 40, 'hidden': 320, 'text_dim': 256, 'focal_alpha': 0.20, 'focal_gamma': 3.0, 'patience': 12, 'tag': 'v4_focal3'},
        # 低 lr 长时间训练
        {'lr': 2e-4, 'bs': 64, 'epochs': 60, 'hidden': 384, 'text_dim': 320, 'focal_alpha': 0.25, 'focal_gamma': 2.0, 'patience': 15, 'tag': 'v4_deep'},
    ]

    # 加载 V3 最佳作为基线
    prev_best_auc = 0.6466
    prev_best_tag = 'v3_wide'
    print(f"  V3 最佳 AUC: {prev_best_auc:.4f} ({prev_best_tag})")

    results = []
    best_auc = prev_best_auc
    best_cfg = None

    for i, hp in enumerate(hp_grid):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(hp_grid)}] {hp['tag']}: lr={hp['lr']}, bs={hp['bs']}, "
              f"ep={hp['epochs']}, h={hp['hidden']}, td={hp['text_dim']}, γ={hp['focal_gamma']}")
        print(f"{'='*60}")

        set_seed(SEED)
        model = KWSpotterV4(
            hidden=hp['hidden'],
            vocab_size=len(vocab),
            text_dim=hp['text_dim']
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {n_params/1e6:.2f}M")

        v_auc, best_ep = train_full_v4(model, train_ds, val_ds, hp, DEVICE)

        results.append({'tag': hp['tag'], 'auc': v_auc, 'best_ep': best_ep, 'hp': hp})
        print(f"  Result: AUC={v_auc:.4f} @ epoch {best_ep}")

        if v_auc > best_auc:
            best_auc = v_auc
            best_cfg = hp
            model_path = OUT / f"best_{hp['tag']}.pt"
            shutil.copy(model_path, OUT / "best_model_v4.pt")
            print(f"  >>> NEW BEST! AUC={v_auc:.4f}")

    # ---- 4. 汇总 ----
    print(f"\n{'#'*60}")
    print(f"  Loop Complete!")
    results.sort(key=lambda r: r['auc'], reverse=True)
    for i, r in enumerate(results):
        m = " *** BEST ***" if i == 0 else ""
        print(f"  {i+1}. [{r['tag']}] AUC={r['auc']:.4f} ep={r['best_ep']}{m}")
    print(f"  Best AUC: {best_auc:.4f}")
    print(f"  vs V3:   {prev_best_auc:.4f} (Δ={best_auc - prev_best_auc:+.4f})")

    # ---- 5. 生成提交 ----
    print(f"\n{'#'*60}")
    print(f"  生成提交文件...")

    if best_cfg is None:
        print(f"  没有新配置超越 V3, 不到 0.7 目标")
        return best_auc

    best_model = KWSpotterV4(
        hidden=best_cfg['hidden'],
        vocab_size=len(vocab),
        text_dim=best_cfg['text_dim']
    ).to(DEVICE)
    best_model.load_state_dict(torch.load(OUT / "best_model_v4.pt", weights_only=True))
    best_model.eval()

    # 测试集特征
    for split_key, split_type, csv_file in [
        ('test_inset', 'inset', TEST / "eval_seen_enroll_text.csv"),
        ('test_outset', 'outset', TEST / "eval_unseen_enroll_text.csv"),
    ]:
        test_cache = OUT / f"{split_key}_feats_v4.pkl"
        if test_cache.exists():
            print(f"  加载测试缓存: {test_cache}")
            with open(test_cache, 'rb') as f:
                all_data[split_key] = pickle.load(f)
        else:
            print(f"  提取测试特征+文本: {split_key}")
            all_data[split_key] = pre_extract_test_with_text(TEST, split_type, csv_file)
            with open(test_cache, 'wb') as f:
                pickle.dump(all_data[split_key], f)

    inset_preds = predict_test_v4(best_model, all_data['test_inset'], char2idx, DEVICE)
    outset_preds = predict_test_v4(best_model, all_data['test_outset'], char2idx, DEVICE)

    make_csv_v4(inset_preds, outset_preds, SUB / f"submit_v4_auc{best_auc:.4f}.csv")

    # 保存结果
    with open(OUT / "results_v4.json", 'w') as f:
        json.dump({
            'best_auc': best_auc, 'best_cfg': str(best_cfg),
            'vs_v3': prev_best_auc,
            'results': [{'tag': r['tag'], 'auc': r['auc'], 'best_ep': r['best_ep']} for r in results],
        }, f, indent=2)

    print(f"\n  Done! Best AUC={best_auc:.4f}")
    return best_auc

if __name__ == "__main__":
    run_loop()
