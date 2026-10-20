"""
Complete V3 Enhanced — 评估已有模型 + 生成提交
================================================
V3E 训练 7/12 模型后中断。此脚本:
1. 评估所有已保存模型在 val 集上的 AUC
2. 选出 best + top-3
3. 生成 test 预测 (inset + outset)
4. 创建单模型 + ensemble 提交文件
5. 保存 results_v3e.json
"""

import os, sys, json, pickle, shutil
from pathlib import Path

# Fix Windows GBK encoding for emoji/unicode
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
OUT_V3 = BASE / "output_v3"
OUT = BASE / "output_v3e"
SUB = BASE / "submission"
OUT.mkdir(exist_ok=True); SUB.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# V3 模型架构 (必须与 baseline_v3_enhanced.py 完全相同)
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

# ============================================================
# Dataset helpers (pad_collate)
# ============================================================

def pad_collate_fn(batch):
    efs, qfs, labels, pids = [], [], [], []
    for b in batch:
        efs.append(b['enroll_feat'].transpose(0, 1))
        qfs.append(b['query_feat'].transpose(0, 1))
        labels.append(b['label']); pids.append(b['pid'])
    efs = nn.utils.rnn.pad_sequence(efs, batch_first=True).transpose(1, 2)
    qfs = nn.utils.rnn.pad_sequence(qfs, batch_first=True).transpose(1, 2)
    return {'ef': efs, 'qf': qfs, 'label': torch.tensor(labels, dtype=torch.float32), 'pid': pids}

class MemKWDataset(Dataset):
    def __init__(self, data_dict):
        self.items = list(data_dict.items())
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        pid, d = self.items[idx]
        return {'enroll_feat': d['enroll_feat'], 'query_feat': d['query_feat'],
                'label': float(d.get('label', 0.5)), 'pid': pid}

# ============================================================
# 评估
# ============================================================

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

@torch.no_grad()
def predict_test(model, test_data, device):
    model.eval()
    preds = {}
    for pid, d in tqdm(test_data.items(), desc="  Predict"):
        ef = d['enroll_feat'].unsqueeze(0).to(device)
        qf = d['query_feat'].unsqueeze(0).to(device)
        preds[pid] = model.predict(ef, qf).cpu().item()
    return preds

@torch.no_grad()
def predict_ensemble(models, test_data, device):
    all_preds = []
    for model in tqdm(models, desc="  Ensemble models"):
        model.eval()
        preds = {}
        for pid, d in test_data.items():
            ef = d['enroll_feat'].unsqueeze(0).to(device)
            qf = d['query_feat'].unsqueeze(0).to(device)
            preds[pid] = model.predict(ef, qf).cpu().item()
        all_preds.append(preds)
    avg_preds = {}
    for pid in all_preds[0]:
        avg_preds[pid] = np.mean([p[pid] for p in all_preds])
    return avg_preds

def make_csv(inset_preds, outset_preds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            f.write(f"{i},inset,{inset_preds.get(f'pair_{i+1:06d}', 0.5):.6f}\n")
        for i in range(50000):
            f.write(f"{i},outset,{outset_preds.get(f'pair_{i+1:06d}', 0.5):.6f}\n")

# ============================================================
# 模型配置映射 (tag -> hp)
# ============================================================

MODEL_CONFIGS = {
    'v3e_big':       {'hidden': 512, 'dropout': 0.35, 'lr': 3e-4, 'bs': 48,  'epochs': 80,  'wd': 2e-4, 'optimizer': 'adamw', 'scheduler': 'plateau', 'label_smoothing': 0.0},
    'v3e_big_long':  {'hidden': 512, 'dropout': 0.40, 'lr': 2e-4, 'bs': 32,  'epochs': 100, 'wd': 3e-4, 'optimizer': 'adamw', 'scheduler': 'cosine', 'label_smoothing': 0.05},
    'v3e_xl':        {'hidden': 512, 'dropout': 0.40, 'lr': 1e-4, 'bs': 32,  'epochs': 120, 'wd': 3e-4, 'optimizer': 'adamw', 'scheduler': 'cosine', 'label_smoothing': 0.05},
    'v3e_wide':      {'hidden': 384, 'dropout': 0.30, 'lr': 5e-4, 'bs': 64,  'epochs': 80,  'wd': 1e-4, 'optimizer': 'adamw', 'scheduler': 'plateau', 'label_smoothing': 0.0},
    'v3e_wide_sgd':  {'hidden': 384, 'dropout': 0.30, 'lr': 5e-4, 'bs': 64,  'epochs': 80,  'wd': 1e-4, 'optimizer': 'sgd',   'scheduler': 'cosine', 'label_smoothing': 0.0},
    'v3e_wide_ls':   {'hidden': 384, 'dropout': 0.35, 'lr': 3e-4, 'bs': 64,  'epochs': 100, 'wd': 2e-4, 'optimizer': 'adamw', 'scheduler': 'plateau', 'label_smoothing': 0.05},
    'best_model_v3e':{'hidden': 512, 'dropout': 0.35, 'lr': 3e-4, 'bs': 48,  'epochs': 80,  'wd': 2e-4, 'optimizer': 'adamw', 'scheduler': 'plateau', 'label_smoothing': 0.0},  # copy of best
}

# ============================================================
# 主流程
# ============================================================

def main():
    # ---- 1. 加载 V3 特征缓存 ----
    print("\n[Step 1] 加载 V3 特征缓存...")
    cache_path = OUT_V3 / "all_features.pkl"
    with open(cache_path, 'rb') as f:
        all_data = pickle.load(f)
    print(f"  Train: {len(all_data['train'])} | Val: {len(all_data['val'])}")

    # ---- 2. 构建 val dataset ----
    print("\n[Step 2] 构建验证集...")
    val_ds = MemKWDataset(all_data['val'])
    print(f"  Val pairs: {len(val_ds)}")

    # ---- 3. 评估所有已保存模型 ----
    print("\n[Step 3] 评估已保存模型...")
    results = []

    model_files = sorted([f for f in os.listdir(OUT) if f.endswith('.pt') and f.startswith('best_v3e')])

    for model_file in model_files:
        tag = model_file.replace('.pt', '')         # e.g. "best_v3e_big"
        tag = tag.replace('best_', '', 1)            # -> "v3e_big"
        if tag not in MODEL_CONFIGS:
            print(f"  [SKIP] {model_file} -> {tag} (no matching config)")
            continue

        hp = MODEL_CONFIGS[tag]
        print(f"\n  --- {tag} (hidden={hp['hidden']}) ---")

        model = KWSpotterV3(hidden=hp['hidden'], dropout=hp['dropout']).to(DEVICE)
        model_path = OUT / model_file
        state = torch.load(model_path, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        v_auc, _ = evaluate(model, val_ds, bs=64, device=DEVICE)
        results.append({'tag': tag, 'auc': float(v_auc), 'hidden': hp['hidden'], 'model_file': model_file})
        print(f"  AUC = {v_auc:.4f}")

    # Also check best_model_v3e.pt if present
    best_path = OUT / "best_model_v3e.pt"
    if best_path.exists():
        # This is a copy of the best model so far during training
        # We already evaluated all individual models, so we can skip this
        # But let's check if it has a different state
        pass

    # ---- 4. 排序并保存结果 ----
    print("\n[Step 4] 汇总结果...")
    results.sort(key=lambda r: r['auc'], reverse=True)

    prev_best_auc = 0.6466  # V3 baseline
    print(f"\n  {'Rank':<6} {'Tag':<25} {'AUC':<10} {'vs V3':<10}")
    print(f"  {'-'*55}")
    for i, r in enumerate(results):
        delta = r['auc'] - prev_best_auc
        marker = " *** BEST ***" if i == 0 else ""
        print(f"  {i+1:<6} {r['tag']:<25} {r['auc']:<10.4f} {delta:+.4f}   {marker}")

    best_auc = results[0]['auc'] if results else prev_best_auc
    top3 = results[:3]

    # 保存 results_v3e.json
    results_json = {
        'best_auc': best_auc,
        'vs_v3': prev_best_auc,
        'top3': [(r['tag'], r['auc']) for r in top3],
        'results': [{'tag': r['tag'], 'auc': r['auc']} for r in results],
    }
    with open(OUT / "results_v3e.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved to {OUT / 'results_v3e.json'}")

    # ---- 5. 释放 train/val 内存, 加载测试特征 ----
    print("\n[Step 5] 释放内存 + 加载测试特征...")
    del all_data['train'], all_data['val'], val_ds
    import gc; gc.collect()

    # 分批加载测试特征 (避免同时占用 ~31GB)
    test_inset_path = OUT_V3 / "test_inset_feats.pkl"
    test_outset_path = OUT_V3 / "test_outset_feats.pkl"

    print(f"  Loading inset test features ({test_inset_path.stat().st_size / 1e9:.1f} GB)...")
    with open(test_inset_path, 'rb') as f:
        test_inset = pickle.load(f)
    print(f"  Inset: {len(test_inset)} pairs")

    print(f"  Loading outset test features ({test_outset_path.stat().st_size / 1e9:.1f} GB)...")
    with open(test_outset_path, 'rb') as f:
        test_outset = pickle.load(f)
    print(f"  Outset: {len(test_outset)} pairs")

    # ---- 6. 单模型最佳预测 ----
    print("\n[Step 6] Single best model prediction...")
    best_tag = results[0]['tag']
    best_hp = MODEL_CONFIGS[best_tag]
    print(f"  Best model: {best_tag} (AUC={best_auc:.4f}, hidden={best_hp['hidden']})")

    best_model = KWSpotterV3(hidden=best_hp['hidden'], dropout=best_hp['dropout']).to(DEVICE)
    best_model.load_state_dict(torch.load(OUT / results[0]['model_file'], weights_only=True))
    best_model.eval()

    print("  Predicting inset...")
    inset_preds = predict_test(best_model, test_inset, DEVICE)
    print("  Predicting outset...")
    outset_preds = predict_test(best_model, test_outset, DEVICE)

    submit_path = SUB / f"submit_v3e_auc{best_auc:.4f}.csv"
    make_csv(inset_preds, outset_preds, submit_path)
    print(f"  Single model submission: {submit_path}")

    # ---- 7. Ensemble 预测 (top-3) ----
    if len(top3) >= 2:
        print(f"\n[Step 7] Ensemble prediction (top-{len(top3)} models)...")
        ensemble_models = []
        for r in top3:
            hp = MODEL_CONFIGS[r['tag']]
            m = KWSpotterV3(hidden=hp['hidden'], dropout=hp['dropout']).to(DEVICE)
            m.load_state_dict(torch.load(OUT / r['model_file'], weights_only=True))
            m.eval()
            ensemble_models.append(m)
            print(f"  Loaded: {r['tag']} (AUC={r['auc']:.4f})")

        ens_inset = predict_ensemble(ensemble_models, test_inset, DEVICE)
        ens_outset = predict_ensemble(ensemble_models, test_outset, DEVICE)

        ens_path = SUB / f"submit_v3e_ensemble_{len(ensemble_models)}x.csv"
        make_csv(ens_inset, ens_outset, ens_path)
        print(f"  Ensemble submission: {ens_path}")

    # ---- 8. 最终汇总 ----
    print(f"\n{'='*60}")
    print(f"  [DONE] V3 Enhanced complete!")
    print(f"  最佳单模型 AUC: {best_auc:.4f} (vs V3: {best_auc - prev_best_auc:+.4f})")
    top3_str = [(r['tag'], f"{r['auc']:.4f}") for r in top3]
    print(f"  Top-3: {top3_str}")
    print(f"  提交文件:")
    print(f"    {submit_path}")
    if len(top3) >= 2:
        print(f"    {ens_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
