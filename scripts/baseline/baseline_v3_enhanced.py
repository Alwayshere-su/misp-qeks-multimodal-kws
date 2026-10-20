"""
MISP-QEKS Baseline V3 Enhanced — 扩展超参搜索 + 集成
=====================================================
基于 V3 已验证架构 (MFCC CNN + Cross-Attn + BCE), 做:
1. 12 轮扩展超参搜索 (vs 9 轮)
2. 更大模型 (hidden up to 512, vs 384)
3. 更长训练 (up to 120 epochs, vs 80)
4. Label Smoothing + SGD+Momentum 替代
5. 保存 top-3 模型用于后续集成
6. 复用 V3 的特征缓存 (不重复提取)
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
OUT_V3 = BASE / "output_v3"      # 复用 V3 缓存
OUT = BASE / "output_v3e"         # 新输出
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
# V3 架构 (完整复用)
# ============================================================

class AudioEncoderV3(nn.Module):
    def __init__(self, in_c=120, hidden=256, out_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_c, hidden, 7, 2, 3), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 5, 2, 2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 5, 2, 2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout),
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

class KWSpotterV3(nn.Module):
    def __init__(self, hidden=256, dropout=0.3):
        super().__init__()
        self.encoder = AudioEncoderV3(120, hidden, hidden, dropout=dropout*0.5)
        self.match_attn = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=dropout*0.3)
        self.match_norm = nn.LayerNorm(hidden)
        self.cls = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Linear(hidden, 1),
        )

    def forward(self, ef, qf):
        e_global, e_frames = self.encoder(ef)
        q_global, q_frames = self.encoder(qf)
        q_attn, _ = self.match_attn(e_global.unsqueeze(1), q_frames, q_frames)
        q_attn = self.match_norm(q_attn.squeeze(1))
        diff = torch.abs(e_global - q_attn)
        prod = e_global * q_attn
        fused = torch.cat([e_global, q_attn, diff, prod], dim=-1)
        return self.cls(fused)

    def predict(self, ef, qf):
        with torch.no_grad():
            return torch.sigmoid(self.forward(ef, qf)).squeeze(-1)

# Feature extraction (from V3, for noise only)
def extract_mfcc_feats(wav_path, max_sec):
    data, sr = sf.read(str(wav_path), dtype='float32')
    if data.ndim > 1: data = data.mean(axis=1)
    audio = torch.from_numpy(data).float()
    if sr != SR: audio = torchaudio.functional.resample(audio, sr, SR)
    max_s = int(max_sec * SR)
    if len(audio) > max_s: audio = audio[:max_s]
    elif len(audio) < max_s: audio = F.pad(audio, (0, max_s - len(audio)))
    audio = audio.unsqueeze(0)
    mfcc = torchaudio.transforms.MFCC(SR, n_mfcc=40, melkwargs={
        'n_mels': 80, 'n_fft': 512, 'hop_length': 160, 'win_length': 400
    })(audio)
    delta = torchaudio.transforms.ComputeDeltas()
    d1, d2 = delta(mfcc), delta(delta(mfcc))
    return torch.cat([mfcc, d1, d2], dim=1).squeeze(0)

# Dataset (V3 compatible)
class MemKWDataset(Dataset):
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
        if self.augment and self.noise_feats and random.random() < 0.3:
            nf = random.choice(self.noise_feats)
            if nf.shape[1] < qf.shape[1]:
                reps = qf.shape[1] // nf.shape[1] + 1; nf = nf.repeat(1, reps)
            nf = nf[:, :qf.shape[1]]
            snr_db = random.choice([-5, 0, 5])
            signal_pow = (qf**2).mean(); noise_pow = (nf**2).mean()
            if noise_pow > 1e-10:
                k = (signal_pow / noise_pow) * (10**(-snr_db/10))
                k = max(0.01, min(k, 100))
                qf = qf + (k**0.5) * nf
        return {'enroll_feat': ef, 'query_feat': qf, 'label': label, 'pid': pid}

def pad_collate_fn(batch):
    efs, qfs, labels, pids = [], [], [], []
    for b in batch:
        efs.append(b['enroll_feat'].transpose(0, 1))
        qfs.append(b['query_feat'].transpose(0, 1))
        labels.append(b['label']); pids.append(b['pid'])
    efs = nn.utils.rnn.pad_sequence(efs, batch_first=True).transpose(1, 2)
    qfs = nn.utils.rnn.pad_sequence(qfs, batch_first=True).transpose(1, 2)
    return {'ef': efs, 'qf': qfs, 'label': torch.tensor(labels, dtype=torch.float32), 'pid': pids}

def balance_sampler(dataset):
    pos_idx = [i for i, (_, d) in enumerate(dataset.items) if d.get('label', 0) == 1]
    neg_idx = [i for i, (_, d) in enumerate(dataset.items) if d.get('label', 0) == 0]
    if not pos_idx: return None
    w = max(1, len(neg_idx) // len(pos_idx))
    indices = neg_idx + pos_idx * w
    random.shuffle(indices)
    return indices

# Label smoothing BCE
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets_smooth)

# ============================================================
# 训练 (增强版)
# ============================================================

def train_one_epoch(model, ds, optimizer, bs, device, criterion, label_smooth=0.0):
    model.train()
    indices = balance_sampler(ds) or list(range(len(ds)))
    random.shuffle(indices)
    total_loss, n = 0, 0
    preds, labels = [], []

    for i in tqdm(range(0, len(indices), bs), desc="  Train", leave=False):
        batch_idx = indices[i:i+bs]
        batch = pad_collate_fn([ds[idx] for idx in batch_idx])
        ef = batch['ef'].to(device); qf = batch['qf'].to(device)
        y = batch['label'].to(device)

        logit = model(ef, qf)
        if label_smooth > 0:
            y_smooth = y * (1 - label_smooth) + 0.5 * label_smooth
            loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), y_smooth)
        else:
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
    model.eval()
    preds, labels = [], []
    for i in tqdm(range(0, len(ds), bs), desc="  Eval", leave=False):
        batch = pad_collate_fn([ds[idx] for idx in range(i, min(i+bs, len(ds)))])
        ef = batch['ef'].to(device); qf = batch['qf'].to(device)
        prob = model.predict(ef, qf)
        preds.extend(prob.cpu().tolist())
        labels.extend(batch['label'].tolist())
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return auc, preds

def train_full(model, train_ds, val_ds, cfg_hp, device):
    # Optimizer
    wd = cfg_hp.get('weight_decay', 1e-4)
    if cfg_hp.get('optimizer', 'adamw') == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg_hp['lr'], momentum=0.9,
                              weight_decay=wd, nesterov=True)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg_hp['lr'], weight_decay=wd)

    # Scheduler
    sched_type = cfg_hp.get('scheduler', 'plateau')
    if sched_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=cfg_hp['lr'] * 0.01)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max', patience=8, factor=0.5)

    ls = cfg_hp.get('label_smoothing', 0.0)
    patience_cfg = cfg_hp.get('patience', 15)
    best_auc, best_ep, patience = 0.5, 0, 0

    for ep in range(1, cfg_hp['epochs'] + 1):
        t_loss, t_auc = train_one_epoch(model, train_ds, optimizer, cfg_hp['bs'], device,
                                         None, label_smooth=ls)
        v_auc, _ = evaluate(model, val_ds, cfg_hp['bs'], device)

        if sched_type == 'plateau':
            scheduler.step(v_auc)
        else:
            scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Ep {ep:4d}: loss={t_loss:.4f} train_AUC={t_auc:.4f} val_AUC={v_auc:.4f} lr={lr_now:.2e}")

        if v_auc > best_auc:
            best_auc = v_auc; best_ep = ep; patience = 0
            torch.save(model.state_dict(), OUT / f"best_{cfg_hp['tag']}.pt")
        else:
            patience += 1

        if patience >= patience_cfg:
            print(f"  早停 @ ep {ep}")
            break

    return best_auc, best_ep

# ============================================================
# 预测测试集 (V3 兼容)
# ============================================================

@torch.no_grad()
def predict_test(model, test_data, device):
    model.eval()
    preds = {}
    for pid, d in tqdm(test_data.items(), desc="  Predict"):
        ef = d['enroll_feat'].unsqueeze(0).to(device)
        qf = d['query_feat'].unsqueeze(0).to(device)
        preds[pid] = model.predict(ef, qf).cpu().item()
    return preds

def make_csv(inset_preds, outset_preds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            f.write(f"{i},inset,{inset_preds.get(f'pair_{i+1:06d}', 0.5):.6f}\n")
        for i in range(50000):
            f.write(f"{i},outset,{outset_preds.get(f'pair_{i+1:06d}', 0.5):.6f}\n")

# ============================================================
# 集成预测
# ============================================================

@torch.no_grad()
def predict_ensemble(models, test_data, device):
    """对多个模型做简单平均集成"""
    all_preds = []
    for model in tqdm(models, desc="  Ensemble models"):
        model.eval()
        preds = {}
        for pid, d in test_data.items():
            ef = d['enroll_feat'].unsqueeze(0).to(device)
            qf = d['query_feat'].unsqueeze(0).to(device)
            preds[pid] = model.predict(ef, qf).cpu().item()
        all_preds.append(preds)

    # 简单平均
    avg_preds = {}
    for pid in all_preds[0]:
        avg_preds[pid] = np.mean([p[pid] for p in all_preds])
    return avg_preds

# ============================================================
# 主 Loop
# ============================================================

def run_loop():
    print(f"Device: {DEVICE}")

    # ---- 1. 加载 V3 特征缓存 (复用!) ----
    print("\n[Phase 1] 加载 V3 特征缓存...")
    cache_path = OUT_V3 / "all_features.pkl"
    with open(cache_path, 'rb') as f:
        all_data = pickle.load(f)
    print(f"  已加载: {len(all_data['train'])} train, {len(all_data['val'])} val pairs")

    # 噪声
    noise_feats = []
    noise_cache = OUT_V3 / "noise_feats.pkl"
    if noise_cache.exists():
        with open(noise_cache, 'rb') as f:
            noise_feats = pickle.load(f)
    print(f"  噪声: {len(noise_feats)}")

    # ---- 2. 数据集 ----
    print("\n[Phase 2] 构建数据集...")
    train_ds = MemKWDataset(all_data['train'], augment=True, noise_feats=noise_feats)
    val_ds = MemKWDataset(all_data['val'], augment=False)
    pos = sum(1 for _, d in train_ds.items if d.get('label', 0) == 1)
    print(f"  Train: {len(train_ds)} ({pos} pos) | Val: {len(val_ds)}")

    # ---- 3. 扩展超参搜索 (12 配置) ----
    print("\n[Phase 3] 扩展超参数搜索 Loop...")

    hp_grid = [
        # ---- Block 1: 大模型 ----
        {'lr': 3e-4, 'bs': 48,  'epochs': 80,  'hidden': 512, 'dropout': 0.35, 'wd': 2e-4, 'patience': 15, 'optimizer': 'adamw', 'scheduler': 'plateau', 'label_smoothing': 0.0, 'tag': 'v3e_big'},
        {'lr': 2e-4, 'bs': 32,  'epochs': 100, 'hidden': 512, 'dropout': 0.40, 'wd': 3e-4, 'patience': 20, 'optimizer': 'adamw', 'scheduler': 'cosine', 'label_smoothing': 0.05, 'tag': 'v3e_big_long'},
        {'lr': 1e-4, 'bs': 32,  'epochs': 120, 'hidden': 512, 'dropout': 0.40, 'wd': 3e-4, 'patience': 25, 'optimizer': 'adamw', 'scheduler': 'cosine', 'label_smoothing': 0.05, 'tag': 'v3e_xl'},

        # ---- Block 2: 中等模型 ----
        {'lr': 5e-4, 'bs': 64,  'epochs': 80,  'hidden': 384, 'dropout': 0.30, 'wd': 1e-4, 'patience': 15, 'optimizer': 'adamw', 'scheduler': 'plateau', 'label_smoothing': 0.0, 'tag': 'v3e_wide'},
        {'lr': 5e-4, 'bs': 64,  'epochs': 80,  'hidden': 384, 'dropout': 0.30, 'wd': 1e-4, 'patience': 15, 'optimizer': 'sgd', 'scheduler': 'cosine', 'label_smoothing': 0.0, 'tag': 'v3e_wide_sgd'},
        {'lr': 3e-4, 'bs': 64,  'epochs': 100, 'hidden': 384, 'dropout': 0.35, 'wd': 2e-4, 'patience': 18, 'optimizer': 'adamw', 'scheduler': 'plateau', 'label_smoothing': 0.05, 'tag': 'v3e_wide_ls'},

        # ---- Block 3: 高 lr + 长训练 ----
        {'lr': 1e-3, 'bs': 64,  'epochs': 70,  'hidden': 320, 'dropout': 0.30, 'wd': 1e-4, 'patience': 12, 'optimizer': 'adamw', 'scheduler': 'cosine', 'label_smoothing': 0.05, 'tag': 'v3e_highlr'},
        {'lr': 5e-4, 'bs': 96,  'epochs': 60,  'hidden': 320, 'dropout': 0.25, 'wd': 1e-4, 'patience': 12, 'optimizer': 'adamw', 'scheduler': 'plateau', 'label_smoothing': 0.0, 'tag': 'v3e_fast'},

        # ---- Block 4: 低 lr + 深训练 ----
        {'lr': 2e-4, 'bs': 48,  'epochs': 120, 'hidden': 320, 'dropout': 0.35, 'wd': 2e-4, 'patience': 20, 'optimizer': 'adamw', 'scheduler': 'cosine', 'label_smoothing': 0.05, 'tag': 'v3e_deep'},
        {'lr': 1e-4, 'bs': 48,  'epochs': 100, 'hidden': 320, 'dropout': 0.35, 'wd': 2e-4, 'patience': 20, 'optimizer': 'sgd', 'scheduler': 'cosine', 'label_smoothing': 0.1, 'tag': 'v3e_sgd_lowlr'},

        # ---- Block 5: 正则化探索 ----
        {'lr': 5e-4, 'bs': 64,  'epochs': 80,  'hidden': 320, 'dropout': 0.40, 'wd': 5e-4, 'patience': 18, 'optimizer': 'adamw', 'scheduler': 'plateau', 'label_smoothing': 0.1, 'tag': 'v3e_heavyreg'},
        {'lr': 3e-4, 'bs': 64,  'epochs': 90,  'hidden': 320, 'dropout': 0.25, 'wd': 5e-5, 'patience': 15, 'optimizer': 'adamw', 'scheduler': 'cosine', 'label_smoothing': 0.0, 'tag': 'v3e_lightreg'},
    ]

    prev_best_auc = 0.6466
    prev_best_tag = 'v3_wide'
    print(f"  V3 best AUC: {prev_best_auc:.4f} ({prev_best_tag})")

    results = []
    best_auc = prev_best_auc
    best_cfg = None
    top_models = []  # (auc, tag) for ensemble

    for i, hp in enumerate(hp_grid):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(hp_grid)}] {hp['tag']}: lr={hp['lr']}, bs={hp['bs']}, "
              f"ep={hp['epochs']}, h={hp['hidden']}, do={hp['dropout']}, "
              f"opt={hp['optimizer']}, ls={hp['label_smoothing']}")
        print(f"{'='*60}")

        set_seed(SEED)
        model = KWSpotterV3(hidden=hp['hidden'], dropout=hp['dropout']).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {n_params/1e6:.2f}M")

        v_auc, best_ep = train_full(model, train_ds, val_ds, hp, DEVICE)
        results.append({'tag': hp['tag'], 'auc': v_auc, 'best_ep': best_ep})
        print(f"  Result: AUC={v_auc:.4f} @ epoch {best_ep}")

        if v_auc > best_auc:
            best_auc = v_auc
            best_cfg = hp
            shutil.copy(OUT / f"best_{hp['tag']}.pt", OUT / "best_model_v3e.pt")
            delta = v_auc - prev_best_auc
            print(f"  >>> NEW BEST! AUC={v_auc:.4f} (vs V3: {'+' if delta>0 else ''}{delta:+.4f})")

        # 跟踪 top-3 模型
        top_models.append((v_auc, hp['tag'], hp))
        top_models.sort(key=lambda x: x[0], reverse=True)
        top_models = top_models[:3]

    # ---- 4. 汇总 ----
    print(f"\n{'#'*60}")
    print(f"  Loop Complete!")
    results.sort(key=lambda r: r['auc'], reverse=True)
    for i, r in enumerate(results):
        m = " *** BEST ***" if i == 0 else ""
        print(f"  {i+1}. [{r['tag']}] AUC={r['auc']:.4f} ep={r['best_ep']}{m}")
    print(f"  Best AUC: {best_auc:.4f} (vs V3: {best_auc-prev_best_auc:+.4f})")

    # ---- 5. Top-3 模型集成 ----
    print(f"\n{'#'*60}")
    print(f"  Top-3 模型集成:")
    for rank, (auc, tag, _) in enumerate(top_models, 1):
        print(f"    {rank}. {tag}: AUC={auc:.4f}")

    # 加载测试特征
    print(f"\n  加载测试特征...")
    for split_key in ['test_inset', 'test_outset']:
        test_cache = OUT_V3 / f"{split_key}_feats.pkl"
        if all_data.get(split_key) is None and test_cache.exists():
            print(f"    加载 {test_cache}")
            with open(test_cache, 'rb') as f:
                all_data[split_key] = pickle.load(f)

    # 单模型最佳预测
    if best_cfg is not None:
        print(f"\n  单模型最佳预测 ({best_cfg['tag']})...")
        best_model = KWSpotterV3(hidden=best_cfg['hidden']).to(DEVICE)
        best_model.load_state_dict(torch.load(OUT / "best_model_v3e.pt", weights_only=True))
        best_model.eval()
        inset_preds = predict_test(best_model, all_data['test_inset'], DEVICE)
        outset_preds = predict_test(best_model, all_data['test_outset'], DEVICE)
        make_csv(inset_preds, outset_preds, SUB / f"submit_v3e_auc{best_auc:.4f}.csv")

    # Ensemble prediction (如果 top-3 都有保存的模型)
    if len(top_models) >= 2:
        print(f"\n  集成预测 (top-{len(top_models)} models)...")
        ensemble_models = []
        for _, tag, hp_cfg in top_models:
            m = KWSpotterV3(hidden=hp_cfg['hidden']).to(DEVICE)
            model_path = OUT / f"best_{tag}.pt"
            if model_path.exists():
                m.load_state_dict(torch.load(model_path, weights_only=True))
                m.eval()
                ensemble_models.append(m)
        if len(ensemble_models) >= 2:
            ens_inset = predict_ensemble(ensemble_models, all_data['test_inset'], DEVICE)
            ens_outset = predict_ensemble(ensemble_models, all_data['test_outset'], DEVICE)
            make_csv(ens_inset, ens_outset, SUB / f"submit_v3e_ensemble_{len(ensemble_models)}x.csv")
            print(f"    集成提交已生成 ({len(ensemble_models)} 模型)")

    # 保存结果
    with open(OUT / "results_v3e.json", 'w') as f:
        json.dump({
            'best_auc': best_auc, 'vs_v3': prev_best_auc,
            'top3': [(auc, tag) for auc, tag, _ in top_models],
            'results': [{'tag': r['tag'], 'auc': r['auc'], 'best_ep': r['best_ep']} for r in results],
        }, f, indent=2)

    print(f"\n  Done!")
    return best_auc

if __name__ == "__main__":
    run_loop()
