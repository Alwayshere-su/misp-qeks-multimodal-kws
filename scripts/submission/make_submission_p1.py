"""
Phase 1 提交生成 — whisper 特征 + MatcherV1
============================================
用法:
  python make_submission_p1.py --model output_p1/best_s2u.pt --out submission/submit_p1_s2u.csv
  python make_submission_p1.py --model output_p1/best_full.pt --out submission/submit_p1_full.csv
"""
import os, sys, pickle, argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_whisper_p1 import MatcherV1, encode_text, CHAR2IDX

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
FEATS = BASE / "output_whisper"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_split(model, data, bs=1024):
    model.eval()
    pids = sorted(data.keys())
    preds = {}
    for i in range(0, len(pids), bs):
        chunk = pids[i:i + bs]
        e = torch.stack([torch.tensor(data[p]['enroll_emb'], dtype=torch.float32) for p in chunk]).to(DEVICE)
        q = torch.stack([torch.tensor(data[p]['query_emb'], dtype=torch.float32) for p in chunk]).to(DEVICE)
        t = torch.stack([encode_text(data[p]['enroll_txt']) for p in chunk]).to(DEVICE)
        logit = model(e, q, t)
        prob = torch.sigmoid(logit).cpu().flatten().tolist()
        for p, pr in zip(chunk, prob):
            preds[p] = pr
    return preds


def make_csv(inset_preds, outset_preds, path):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},inset,{inset_preds.get(pid, 0.5):.6f}\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},outset,{outset_preds.get(pid, 0.5):.6f}\n")
    print(f"  -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='output_p1/best_s2u.pt')
    parser.add_argument('--out', default='submission/submit_p1_s2u.csv')
    args = parser.parse_args()

    model = MatcherV1().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True))
    model.eval()
    print(f"Loaded {args.model}")

    print("Loading test features...")
    inset = pickle.load(open(FEATS / "test_inset.pkl", 'rb'))
    outset = pickle.load(open(FEATS / "test_outset.pkl", 'rb'))
    print(f"  inset: {len(inset)}, outset: {len(outset)}")

    print("Predicting inset...")
    inset_preds = predict_split(model, inset)
    print("Predicting outset...")
    outset_preds = predict_split(model, outset)

    make_csv(inset_preds, outset_preds, args.out)

    # 预测分布检查
    allv = list(inset_preds.values()) + list(outset_preds.values())
    print(f"  pred stats: min={min(allv):.4f} max={max(allv):.4f} mean={np.mean(allv):.4f} "
          f"std={np.std(allv):.4f} n50={sum(1 for v in allv if abs(v-0.5)<0.001)}")


if __name__ == '__main__':
    main()
