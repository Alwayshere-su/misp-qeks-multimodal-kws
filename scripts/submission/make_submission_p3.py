"""
提交生成 — MatcherV3 (全局 + 帧级特征 + 文本)
用法:
  python make_submission_p3.py --model output_p3/best_full.pt --out submission/submit_p3_full.csv
  python make_submission_p3.py --model output_p3/best_s2u.pt --out submission/submit_p3_s2u.csv
"""
import sys, pickle, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_whisper_p3 import MatcherV3, encode_text

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
FEATS = BASE / "output_whisper"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_split(model, data, bs=512):
    model.eval()
    pids = sorted(data.keys())
    preds = {}
    for i in range(0, len(pids), bs):
        chunk = pids[i:i + bs]
        e = torch.stack([torch.tensor(data[p]['enroll_emb'], dtype=torch.float32) for p in chunk]).to(DEVICE)
        ef = torch.stack([torch.tensor(data[p]['enroll_frames'], dtype=torch.float32) for p in chunk]).to(DEVICE)
        q = torch.stack([torch.tensor(data[p]['query_emb'], dtype=torch.float32) for p in chunk]).to(DEVICE)
        qf = torch.stack([torch.tensor(data[p]['query_frames'], dtype=torch.float32) for p in chunk]).to(DEVICE)
        t = torch.stack([encode_text(data[p]['enroll_txt']) for p in chunk]).to(DEVICE)
        logit = model(e, ef, q, qf, t)
        prob = torch.sigmoid(logit).cpu().flatten().tolist()
        for p, pr in zip(chunk, prob):
            preds[p] = pr
    return preds


def make_csv(inset_preds, outset_preds, path):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            f.write(f"{i},inset,{inset_preds.get(f'pair_{i+1:06d}', 0.5):.6f}\n")
        for i in range(50000):
            f.write(f"{i},outset,{outset_preds.get(f'pair_{i+1:06d}', 0.5):.6f}\n")
    print(f"  -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='output_p3/best_full.pt')
    parser.add_argument('--out', default='submission/submit_p3_full.csv')
    args = parser.parse_args()

    model = MatcherV3().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True))
    model.eval()
    print(f"Loaded {args.model}")

    print("Loading test features...")
    inset = pickle.load(open(FEATS / "test_inset.pkl", 'rb'))
    outset = pickle.load(open(FEATS / "test_outset.pkl", 'rb'))
    print(f"  inset: {len(inset)}, outset: {len(outset)}")

    inset_preds = predict_split(model, inset)
    outset_preds = predict_split(model, outset)
    make_csv(inset_preds, outset_preds, args.out)

    allv = list(inset_preds.values()) + list(outset_preds.values())
    print(f"  pred stats: mean={np.mean(allv):.4f} std={np.std(allv):.4f}")


if __name__ == '__main__':
    main()
