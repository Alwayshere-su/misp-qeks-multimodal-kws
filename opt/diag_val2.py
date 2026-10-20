"""Evaluate current p4 model + text match + frame match on val splits."""
import pickle, sys
from pathlib import Path
import numpy as np
import torch

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
FEATS = BASE / "output_whisper"
LIP = BASE / "output_lip"
sys.path.insert(0, str(BASE))
from train_whisper_p4 import MatcherV4, encode_text


def rank_auc(y, s):
    s = np.asarray(s); y = np.asarray(y)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = pickle.load(open(FEATS / "val_feats.pkl", "rb"))
    lip = pickle.load(open(LIP / "val_lip.pkl", "rb"))

    model = MatcherV4().to(DEVICE)
    model.load_state_dict(torch.load(BASE / "output_p4/best_full.pt", map_location="cpu", weights_only=True))
    model.eval()

    for name in ["dev_seen", "dev_unseen"]:
        pids = sorted(d[name].keys())
        y, p_model, p_text, p_frame = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(pids), 512):
                chunk = pids[i:i + 512]
                ea = torch.stack([torch.tensor(d[name][p]["enroll_emb"]) for p in chunk]).to(DEVICE)
                qa = torch.stack([torch.tensor(d[name][p]["query_emb"]) for p in chunk]).to(DEVICE)
                ev = torch.stack([torch.tensor(lip[name][p]["enroll_video"]) for p in chunk]).to(DEVICE)
                qv = torch.stack([torch.tensor(lip[name][p]["query_video"]) for p in chunk]).to(DEVICE)
                t = torch.stack([encode_text(d[name][p]["enroll_txt"]) for p in chunk]).to(DEVICE)
                logit = model(ea, qa, ev, qv, t)
                p_model.extend(torch.sigmoid(logit).cpu().flatten().tolist())
        for p in pids:
            it = d[name][p]
            et = it["enroll_txt"].lower().strip()
            qt = it["query_txt"].lower().strip()
            p_text.append(1.0 if et and qt and et in qt else 0.0)
            ef = it["enroll_frames"].astype(np.float32)
            qf = it["query_frames"].astype(np.float32)
            ef = ef / (np.linalg.norm(ef, axis=1, keepdims=True) + 1e-9)
            qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-9)
            sim = ef @ qf.T
            p_frame.append(float(sim.max(0).mean()))
            y.append(it["label"])
        y = np.array(y)
        print(f"=== {name} (n={len(y)}, pos={y.mean():.3f}) ===")
        print(f"  p4 model AUC       = {rank_auc(y, p_model):.4f}")
        print(f"  text exact-match   = {rank_auc(y, p_text):.4f}  (pos hit={np.mean([p_text[i]==1 and y[i]==1 for i in range(len(y))]):.3f})")
        print(f"  frame mean-of-max  = {rank_auc(y, p_frame):.4f}")


if __name__ == "__main__":
    main()
