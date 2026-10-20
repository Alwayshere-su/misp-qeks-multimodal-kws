"""
V3E Predict Only — 用已评估的 best+top3 模型生成提交
=======================================================
直接读取 results_v3e.json, 无需重新评估模型。
"""

import os, sys, json, pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import gc

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
OUT_V3 = BASE / "output_v3"
OUT = BASE / "output_v3e"
SUB = BASE / "submission"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# V3 Model Architecture
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
# Tag -> HP config
# ============================================================
MODEL_CONFIGS = {
    'v3e_big':       {'hidden': 512, 'dropout': 0.35},
    'v3e_big_long':  {'hidden': 512, 'dropout': 0.40},
    'v3e_xl':        {'hidden': 512, 'dropout': 0.40},
    'v3e_wide':      {'hidden': 384, 'dropout': 0.30},
    'v3e_wide_sgd':  {'hidden': 384, 'dropout': 0.30},
    'v3e_wide_ls':   {'hidden': 384, 'dropout': 0.35},
}

# ============================================================
# Prediction helpers
# ============================================================
@torch.no_grad()
def predict_test(model, test_data, device, desc="Predict"):
    model.eval()
    preds = {}
    items = list(test_data.items())
    for pid, d in tqdm(items, desc=desc):
        ef = d['enroll_feat'].unsqueeze(0).to(device)
        qf = d['query_feat'].unsqueeze(0).to(device)
        preds[pid] = model.predict(ef, qf).cpu().item()
    return preds

@torch.no_grad()
def predict_ensemble(models, test_data, device):
    all_preds = []
    for i, model in enumerate(tqdm(models, desc="Ensemble models")):
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
# Main
# ============================================================
def main():
    # 1. Load results
    print("\n[1] Loading results...")
    with open(OUT / "results_v3e.json") as f:
        res = json.load(f)

    best_auc = res['best_auc']
    top3 = res['top3']
    print(f"  Best: {top3[0][0]} AUC={best_auc:.4f}")
    for i, (tag, auc) in enumerate(top3):
        print(f"  Top-{i+1}: {tag} AUC={auc:.4f}")

    # 2. Load test features
    print("\n[2] Loading test features...")
    print("  Loading inset (14.5 GB)...")
    with open(OUT_V3 / "test_inset_feats.pkl", 'rb') as f:
        test_inset = pickle.load(f)
    print(f"  Inset: {len(test_inset)} pairs")

    print("  Loading outset (14.5 GB)...")
    with open(OUT_V3 / "test_outset_feats.pkl", 'rb') as f:
        test_outset = pickle.load(f)
    print(f"  Outset: {len(test_outset)} pairs")

    # 3. Single best model prediction
    print("\n[3] Single best model prediction...")
    best_tag = top3[0][0]
    best_hp = MODEL_CONFIGS[best_tag]
    best_model = KWSpotterV3(hidden=best_hp['hidden'], dropout=best_hp['dropout']).to(DEVICE)
    best_model.load_state_dict(torch.load(OUT / f"best_{best_tag}.pt", weights_only=True))
    best_model.eval()

    inset_preds = predict_test(best_model, test_inset, DEVICE, "Inset")
    outset_preds = predict_test(best_model, test_outset, DEVICE, "Outset")

    submit_path = SUB / f"submit_v3e_auc{best_auc:.4f}.csv"
    make_csv(inset_preds, outset_preds, submit_path)
    print(f"  Saved: {submit_path}")

    # 4. Ensemble prediction
    if len(top3) >= 2:
        print(f"\n[4] Ensemble prediction (top-{len(top3)})...")
        ensemble_models = []
        for tag, auc in top3:
            hp = MODEL_CONFIGS[tag]
            m = KWSpotterV3(hidden=hp['hidden'], dropout=hp['dropout']).to(DEVICE)
            m.load_state_dict(torch.load(OUT / f"best_{tag}.pt", weights_only=True))
            m.eval()
            ensemble_models.append(m)

        ens_inset = predict_ensemble(ensemble_models, test_inset, DEVICE)
        ens_outset = predict_ensemble(ensemble_models, test_outset, DEVICE)

        ens_path = SUB / f"submit_v3e_ensemble_{len(ensemble_models)}x.csv"
        make_csv(ens_inset, ens_outset, ens_path)
        print(f"  Saved: {ens_path}")

    print("\n[DONE] All submissions generated!")

if __name__ == "__main__":
    main()
