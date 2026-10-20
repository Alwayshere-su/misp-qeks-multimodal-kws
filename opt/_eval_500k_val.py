"""本地评估 best_500k.pt 在 val 上的 AUC (归因诊断)
用法: python opt/_eval_500k_val.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

BASE = Path(r"D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "opt"))
from train_500k_matcher import Matcher500K, encode_text, DEVICE  # noqa: E402


@torch.no_grad()
def predict(model, e, q, t, bs=2048):
    model.eval()
    outs = []
    for i in range(0, len(e), bs):
        eb = torch.from_numpy(e[i:i + bs]).float().to(DEVICE)
        qb = torch.from_numpy(q[i:i + bs]).float().to(DEVICE)
        tb = torch.stack([encode_text(x) for x in t[i:i + bs]]).to(DEVICE)
        outs.append(torch.sigmoid(model(eb, qb, tb)).cpu().numpy())
    return np.concatenate(outs)


def main():
    model = Matcher500K().to(DEVICE)
    sd = torch.load(BASE / "opt/out_500k/best_500k.pt", map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()
    print("best_500k.pt 加载成功", flush=True)

    vd = pickle.load(open(BASE / "output_whisper/val_feats.pkl", "rb"))
    for split in ["dev_seen", "dev_unseen"]:
        pids = sorted(vd[split].keys())
        e = np.stack([vd[split][p]["enroll_emb"] for p in pids]).astype(np.float32)
        q = np.stack([vd[split][p]["query_emb"] for p in pids]).astype(np.float32)
        t = [vd[split][p]["enroll_txt"] for p in pids]
        y = np.array([vd[split][p]["label"] for p in pids])
        pr = predict(model, e, q, t)
        print(f"val {split}: AUC={roc_auc_score(y, pr):.6f} n={len(y)} "
              f"pos={y.mean():.3f} pred_mean={pr.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
