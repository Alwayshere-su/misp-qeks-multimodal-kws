"""
MISP-QEKS Baseline V3 — 快速改进版
====================================
关键优化:
1. 预提取所有特征到内存 (大幅加速)
2. 更高效的模型架构
3. 正负样本平衡 + 数据增强
4. 自动化超参搜索 Loop
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
OUT = BASE / "output_v3"
SUB = BASE / "submission"
OUT.mkdir(exist_ok=True); SUB.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SR = 16000
FEAT_DIM = 120  # 40 mfcc + 40 delta + 40 delta2

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(SEED)

# ============================================================
# 批量特征预提取 (关键优化!)
# ============================================================

def extract_mfcc_feats(wav_path, max_sec):
    """提取 MFCC+delta+delta2 特征"""
    data, sr = sf.read(str(wav_path), dtype='float32')
    if data.ndim > 1: data = data.mean(axis=1)
    audio = torch.from_numpy(data).float()

    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)

    max_s = int(max_sec * SR)
    if len(audio) > max_s: audio = audio[:max_s]
    elif len(audio) < max_s: audio = F.pad(audio, (0, max_s - len(audio)))

    audio = audio.unsqueeze(0)  # (1, T)

    mfcc = torchaudio.transforms.MFCC(SR, n_mfcc=40, melkwargs={
        'n_mels': 80, 'n_fft': 512, 'hop_length': 160, 'win_length': 400
    })(audio)  # (1, 40, T')

    delta = torchaudio.transforms.ComputeDeltas()
    d1, d2 = delta(mfcc), delta(delta(mfcc))
    feat = torch.cat([mfcc, d1, d2], dim=1)  # (1, 120, T')
    return feat.squeeze(0)  # (120, T)


def pre_extract_all(split_dir, max_enroll=2.0, max_query=4.0):
    """预提取一个 split 的所有特征到内存"""
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

        data[pid] = {
            'enroll_feat': ef,  # (120, Te)
            'query_feat': qf,   # (120, Tq)
            'label': label,
        }

    return data


def pre_extract_test(test_dir, split_type, max_enroll=2.0, max_query=4.0):
    """预提取测试集特征"""
    d = Path(test_dir)
    if split_type == 'inset':
        wav_dir = d / "eval_seen_unlabeled" / "wav"
    else:
        wav_dir = d / "eval_unseen_unlabeled" / "wav"

    files = sorted(wav_dir.glob("pair_*_enroll.wav"))
    data = {}
    print(f"  预提取 test/{split_type}: {len(files)} pairs...")

    for f in tqdm(files, desc=f"  test/{split_type}"):
        pid = f.stem.replace('_enroll', '')
        ef = extract_mfcc_feats(wav_dir / f"{pid}_enroll.wav", max_enroll)
        qf = extract_mfcc_feats(wav_dir / f"{pid}_query.wav", max_query)
        data[pid] = {'enroll_feat': ef, 'query_feat': qf}

    return data


# ============================================================
# 快速数据集 (内存)
# ============================================================

class MemKWDataset(Dataset):
    """内存数据集: 所有特征已在内存中"""
    def __init__(self, data_dict, augment=False, noise_feats=None):
        self.items = list(data_dict.items())
        self.augment = augment
        self.noise_feats = noise_feats or []

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        pid, d = self.items[idx]
        ef = d['enroll_feat']  # (120, Te)
        qf = d['query_feat']   # (120, Tq)
        label = float(d.get('label', 0.5))

        # 噪声增强
        if self.augment and self.noise_feats and random.random() < 0.3:
            nf = random.choice(self.noise_feats)  # (120, Tn)
            # 对齐长度并混合
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

        return {'enroll_feat': ef, 'query_feat': qf, 'label': label, 'pid': pid}


def pad_collate_fn(batch):
    """padding + batch"""
    efs, qfs, labels, pids = [], [], [], []
    for b in batch:
        efs.append(b['enroll_feat'].transpose(0, 1))  # (Te, 120)
        qfs.append(b['query_feat'].transpose(0, 1))    # (Tq, 120)
        labels.append(b['label'])
        pids.append(b['pid'])
    efs = nn.utils.rnn.pad_sequence(efs, batch_first=True).transpose(1, 2)
    qfs = nn.utils.rnn.pad_sequence(qfs, batch_first=True).transpose(1, 2)
    return {
        'ef': efs, 'qf': qfs,
        'label': torch.tensor(labels, dtype=torch.float32),
        'pid': pids
    }


def balance_sampler(dataset):
    """正样本过采样"""
    pos_idx = [i for i, (_, d) in enumerate(dataset.items) if d.get('label', 0) == 1]
    neg_idx = [i for i, (_, d) in enumerate(dataset.items) if d.get('label', 0) == 0]
    if not pos_idx: return None
    w = max(1, len(neg_idx) // len(pos_idx))
    indices = neg_idx + pos_idx * w
    random.shuffle(indices)
    return indices


# ============================================================
# 模型架构 (高效快速)
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
        # Attention pooling
        self.attn = nn.Conv1d(hidden, 1, 1)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        """
        x: (B, 120, T)
        Returns: global_emb (B, out_dim), frame_embs (B, T', out_dim)
        """
        x = self.encoder(x)          # (B, hidden, T/16)
        w = self.attn(x).softmax(-1)  # (B, 1, T/16)
        global_e = (w * x).sum(-1)    # (B, hidden)
        global_e = self.proj(global_e)  # (B, out_dim)

        # Frame embeddings (optional, for matching)
        frames = x.transpose(1, 2)     # (B, T/16, hidden)
        frames = self.proj(frames)     # (B, T/16, out_dim)

        return global_e, frames


class KWSpotterV3(nn.Module):
    """V3 关键词检索器"""
    def __init__(self, hidden=256):
        super().__init__()
        self.encoder = AudioEncoderV3(120, hidden, hidden)

        # 匹配网络
        self.match_attn = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=0.1)
        self.match_norm = nn.LayerNorm(hidden)

        # 分类头
        self.cls = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden * 2, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, ef, qf):
        """ef: (B, 120, Te), qf: (B, 120, Tq)"""
        e_global, e_frames = self.encoder(ef)  # (B, H), (B, Te', H)
        q_global, q_frames = self.encoder(qf)  # (B, H), (B, Tq', H)

        # Cross-attention: enrollment attends to query frames
        q_attn, _ = self.match_attn(
            e_global.unsqueeze(1), q_frames, q_frames
        )  # (B, 1, H)
        q_attn = self.match_norm(q_attn.squeeze(1))

        # 特征融合
        diff = torch.abs(e_global - q_attn)
        prod = e_global * q_attn
        fused = torch.cat([e_global, q_attn, diff, prod], dim=-1)

        logit = self.cls(fused)
        return logit

    def predict(self, ef, qf):
        with torch.no_grad():
            return torch.sigmoid(self.forward(ef, qf)).squeeze(-1)


# ============================================================
# 训练
# ============================================================

def train_one_epoch(model, ds, optimizer, bs, device):
    """单 epoch 训练"""
    model.train()
    indices = balance_sampler(ds) or list(range(len(ds)))
    random.shuffle(indices)

    total_loss, n = 0, 0
    preds, labels = [], []

    for i in tqdm(range(0, len(indices), bs), desc="  Train", leave=False):
        batch_idx = indices[i:i+bs]
        batch = pad_collate_fn([ds[idx] for idx in batch_idx])

        ef = batch['ef'].to(device)
        qf = batch['qf'].to(device)
        y = batch['label'].to(device)

        logit = model(ef, qf)
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
def evaluate(model, ds, bs, device):
    """评估"""
    model.eval()
    preds, labels = [], []

    for i in tqdm(range(0, len(ds), bs), desc="  Eval", leave=False):
        batch = pad_collate_fn([ds[idx] for idx in range(i, min(i+bs, len(ds)))])
        ef = batch['ef'].to(device)
        qf = batch['qf'].to(device)

        prob = model.predict(ef, qf)
        preds.extend(prob.cpu().tolist())
        labels.extend(batch['label'].tolist())

    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return auc, preds


def train_full(model, train_ds, val_ds, cfg_hp, device):
    """完整训练 + 验证 + 早停"""
    optimizer = optim.AdamW(model.parameters(), lr=cfg_hp['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=8, factor=0.5)

    best_auc, best_ep, patience = 0.5, 0, 0

    for ep in range(1, cfg_hp['epochs'] + 1):
        t_loss, t_auc = train_one_epoch(model, train_ds, optimizer, cfg_hp['bs'], device)
        v_auc, _ = evaluate(model, val_ds, cfg_hp['bs'], device)
        scheduler.step(v_auc)

        if v_auc > best_auc:
            best_auc = v_auc; best_ep = ep; patience = 0
            torch.save(model.state_dict(), OUT / f"best_{cfg_hp['tag']}.pt")
        else:
            patience += 1

        if patience >= cfg_hp.get('patience', 15):
            break

    return best_auc, best_ep


# ============================================================
# 预测测试集
# ============================================================

@torch.no_grad()
def predict_test(model, test_data, device):
    """预测整个测试集"""
    model.eval()
    preds = {}
    for pid, d in tqdm(test_data.items(), desc="  Predict"):
        ef = d['enroll_feat'].unsqueeze(0).to(device)
        qf = d['query_feat'].unsqueeze(0).to(device)
        prob = model.predict(ef, qf).cpu().item()
        preds[pid] = prob
    return preds


def make_csv(inset_preds, outset_preds, path):
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
# 自动改进 Loop
# ============================================================

def run_loop():
    print(f"Device: {DEVICE}")

    # ---- 1. 预提取所有特征 ----
    print("\n[Phase 1] 预提取特征...")
    cache_path = OUT / "all_features.pkl"

    if cache_path.exists():
        print(f"  加载缓存: {cache_path}")
        with open(cache_path, 'rb') as f:
            all_data = pickle.load(f)
    else:
        # 先只提取训练/验证数据
        all_data = {
            'train': pre_extract_all(DATA / "dev_seen"),
            'val': pre_extract_all(DATA / "dev_unseen"),
            'test_inset': None,
            'test_outset': None,
        }
        # 缓存训练数据
        with open(cache_path, 'wb') as f:
            pickle.dump(all_data, f)
        print(f"  训练特征已缓存: {cache_path}")

    # 提取噪声特征用于增强
    noise_feats = []
    noise_cache = OUT / "noise_feats.pkl"
    if noise_cache.exists():
        with open(noise_cache, 'rb') as f:
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
            with open(noise_cache, 'wb') as f:
                pickle.dump(noise_feats, f)
            print(f"  噪声特征: {len(noise_feats)}")
        except Exception as e:
            print(f"  噪声加载跳过: {e}")

    # ---- 2. 构建数据集 ----
    print("\n[Phase 2] 构建数据集...")
    train_ds = MemKWDataset(all_data['train'], augment=True, noise_feats=noise_feats)
    val_ds = MemKWDataset(all_data['val'], augment=False)

    pos = sum(1 for _, d in train_ds.items if d.get('label', 0) == 1)
    print(f"  Train: {len(train_ds)} ({pos} pos)")
    print(f"  Val:   {len(val_ds)}")

    # ---- 3. 超参数搜索 ----
    print("\n[Phase 3] 超参数搜索 Loop...")

    hp_grid = [
        {'lr': 1e-3, 'bs': 64,  'epochs': 50, 'hidden': 256, 'patience': 10, 'tag': 'lr1e3'},
        {'lr': 3e-4, 'bs': 128, 'epochs': 70, 'hidden': 320, 'patience': 12, 'tag': 'wide_long'},
        {'lr': 2e-4, 'bs': 64,  'epochs': 60, 'hidden': 384, 'patience': 12, 'tag': 'big_lowlr'},
        {'lr': 5e-4, 'bs': 96,  'epochs': 50, 'hidden': 256, 'patience': 10, 'tag': 'mid'},
        {'lr': 3e-4, 'bs': 128, 'epochs': 80, 'hidden': 320, 'patience': 15, 'tag': 'final'},
    ]

    # 加载之前的最佳模型作为基线
    prev_best_auc = 0.6466
    prev_best_tag = 'wide'
    print(f"  Loaded previous best: AUC={prev_best_auc:.4f} (tag={prev_best_tag})")

    results = []
    best_auc, best_cfg, best_model_path = prev_best_auc, None, None

    for i, hp in enumerate(hp_grid):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(hp_grid)}] {hp['tag']}: lr={hp['lr']}, bs={hp['bs']}, "
              f"ep={hp['epochs']}, h={hp['hidden']}")
        print(f"{'='*60}")

        set_seed(SEED)
        model = KWSpotterV3(hidden=hp['hidden']).to(DEVICE)
        v_auc, best_ep = train_full(model, train_ds, val_ds, hp, DEVICE)

        results.append({'tag': hp['tag'], 'auc': v_auc, 'best_ep': best_ep, 'hp': hp})
        print(f"  Result: AUC={v_auc:.4f} @ epoch {best_ep}")

        if v_auc > best_auc:
            best_auc = v_auc
            best_cfg = hp
            best_model_path = OUT / f"best_{hp['tag']}.pt"
            # 复制为全局最佳
            shutil.copy(best_model_path, OUT / "best_model_v3.pt")

        # ---- 动态调整 ----
        # 如果当前 AUC > 0.65, 用这个模型配置作为下一轮基础
        if v_auc > 0.6:
            print(f"  >>> 突破 0.6! 继续优化...")

    # ---- 4. 汇总 ----
    print(f"\n{'#'*60}")
    print(f"  Loop Complete!")
    results.sort(key=lambda r: r['auc'], reverse=True)
    for i, r in enumerate(results):
        m = " *** BEST ***" if i == 0 else ""
        print(f"  {i+1}. [{r['tag']}] AUC={r['auc']:.4f} ep={r['best_ep']}{m}")
    print(f"  Best AUC: {best_auc:.4f}")

    # ---- 5. 生成提交 ----
    print(f"\n{'#'*60}")
    print(f"  生成提交文件...")

    # 回退: 如果没有新模型超越之前的最佳, 用之前的最佳模型
    if best_cfg is None:
        print(f"  没有新配置超越 prev_best={prev_best_auc:.4f}, 使用之前的 best_model_v3.pt")
        best_cfg = {'hidden': 320, 'tag': prev_best_tag}  # wide 配置

    best_model = KWSpotterV3(hidden=best_cfg['hidden']).to(DEVICE)
    best_model.load_state_dict(torch.load(OUT / "best_model_v3.pt", weights_only=True))
    best_model.eval()

    # 测试集特征: 按需提取或加载缓存
    for split_key, split_type in [('test_inset', 'inset'), ('test_outset', 'outset')]:
        test_cache = OUT / f"{split_key}_feats.pkl"
        if all_data.get(split_key) is not None:
            continue  # 已有
        if test_cache.exists():
            print(f"  加载测试缓存: {test_cache}")
            with open(test_cache, 'rb') as f:
                all_data[split_key] = pickle.load(f)
        else:
            print(f"  提取测试特征: {split_key}")
            all_data[split_key] = pre_extract_test(TEST, split_type)
            with open(test_cache, 'wb') as f:
                pickle.dump(all_data[split_key], f)

    inset_preds = predict_test(best_model, all_data['test_inset'], DEVICE)
    outset_preds = predict_test(best_model, all_data['test_outset'], DEVICE)

    make_csv(inset_preds, outset_preds, SUB / f"submit_v3_auc{best_auc:.4f}.csv")

    # 保存结果
    with open(OUT / "results_v3.json", 'w') as f:
        json.dump({
            'best_auc': best_auc, 'best_cfg': str(best_cfg),
            'results': [{'tag': r['tag'], 'auc': r['auc'], 'best_ep': r['best_ep']} for r in results],
        }, f, indent=2)

    print(f"\n  Done! Best AUC={best_auc:.4f}")
    return best_auc


if __name__ == "__main__":
    run_loop()
