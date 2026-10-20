"""Diagnose val feature properties: why local CV=0.987 but online=0.82755."""
import pickle
from pathlib import Path
import numpy as np

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
FEATS = BASE / "output_whisper"
LIP = BASE / "output_lip"


def cos_auc(E, Q, y):
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    s = (En * Qn).sum(1)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc, s


d = pickle.load(open(FEATS / "val_feats.pkl", "rb"))
lip = pickle.load(open(LIP / "val_lip.pkl", "rb"))

for name in ["dev_seen", "dev_unseen"]:
    items = list(d[name].values())
    E = np.stack([it["enroll_emb"] for it in items])
    Q = np.stack([it["query_emb"] for it in items])
    y = np.array([it["label"] for it in items])
    auc_a, sa = cos_auc(E, Q, y)

    pids = list(d[name].keys())
    vE = np.stack([lip[name][p]["enroll_video"] for p in pids])
    vQ = np.stack([lip[name][p]["query_video"] for p in pids])
    auc_v, sv = cos_auc(vE, vQ, y)

    maxcos = []
    meanmax = []
    labels = []
    for i, pid in enumerate(pids[:3000]):
        ef = d[name][pid]["enroll_frames"].astype(np.float32)
        qf = d[name][pid]["query_frames"].astype(np.float32)
        ef = ef / (np.linalg.norm(ef, axis=1, keepdims=True) + 1e-9)
        qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-9)
        sim = ef @ qf.T
        maxcos.append(sim.max())
        meanmax.append(sim.max(0).mean())
        labels.append(d[name][pid]["label"])
    labels = np.array(labels)
    maxcos = np.array(maxcos); meanmax = np.array(meanmax)

    print(f"=== {name} (n={len(y)}, pos={y.mean():.3f}) ===")
    print(f"  audio global cosine AUC = {auc_a:.4f}")
    print(f"  video global cosine AUC = {auc_v:.4f}")
    print(f"  pos audio cos mean={sa[y==1].mean():.4f}  neg cos mean={sa[y==0].mean():.4f}")
    print(f"  frame max-sim (3000) pos={maxcos[labels==1].mean():.4f} neg={maxcos[labels==0].mean():.4f}")
    print(f"  frame mean-of-max (3000) pos={meanmax[labels==1].mean():.4f} neg={meanmax[labels==0].mean():.4f}")
