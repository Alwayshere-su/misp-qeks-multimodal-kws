"""
提交生成 — MatcherV4 (音频 + 视频 + 文本)
用法: python make_submission_p4.py --model output_p4/best_full.pt --out submission/submit_p4_full.csv
"""
import os, sys, pickle, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_whisper_p4 import MatcherV4, encode_text

BASE = Path(os.environ.get("MISP_ROOT", Path(__file__).resolve().parents[2]))
FEATS = BASE / "output_whisper"
LIP = BASE / "output_lip"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_split(model, data, lip, bs=512):
    model.eval()
    pids = sorted(data.keys())
    preds = {}
    for i in range(0, len(pids), bs):
        chunk = pids[i:i + bs]
        ea = torch.stack([torch.tensor(data[p]['enroll_emb'], dtype=torch.float32) for p in chunk]).to(DEVICE)
        qa = torch.stack([torch.tensor(data[p]['query_emb'], dtype=torch.float32) for p in chunk]).to(DEVICE)
        ev = torch.stack([torch.tensor(lip[p]['enroll_video'], dtype=torch.float32) for p in chunk]).to(DEVICE)
        qv = torch.stack([torch.tensor(lip[p]['query_video'], dtype=torch.float32) for p in chunk]).to(DEVICE)
        t = torch.stack([encode_text(data[p]['enroll_txt']) for p in chunk]).to(DEVICE)
        logit = model(ea, qa, ev, qv, t)
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
    parser.add_argument('--model', default='output_p4/best_full.pt')
    parser.add_argument('--out', default='submission/submit_p4_full.csv')
    args = parser.parse_args()

    model = MatcherV4().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True))
    model.eval()
    print(f"Loaded {args.model}")

    print("Loading test features (audio + video)...")
    inset = pickle.load(open(FEATS / "test_inset.pkl", 'rb'))
    outset = pickle.load(open(FEATS / "test_outset.pkl", 'rb'))
    lip_inset = pickle.load(open(LIP / "test_lip_inset.pkl", 'rb'))
    lip_outset = pickle.load(open(LIP / "test_lip_outset.pkl", 'rb'))
    print(f"  inset: {len(inset)}/{len(lip_inset)}, outset: {len(outset)}/{len(lip_outset)}")

    inset_preds = predict_split(model, inset, lip_inset)
    outset_preds = predict_split(model, outset, lip_outset)
    make_csv(inset_preds, outset_preds, args.out)

    allv = list(inset_preds.values()) + list(outset_preds.values())
    print(f"  pred stats: mean={np.mean(allv):.4f} std={np.std(allv):.4f}")


if __name__ == '__main__':
    main()
