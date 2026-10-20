"""
p4 声学模型对 val 全量打分 (用于与 ASR 分数融合评估)
用法: python score_val_acoustic.py --model output_p4/best_full.pt --out output_p4/val_scores_full.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_whisper_p4 import MatcherV4, encode_text

BASE = Path(r"D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='output_p4/best_full.pt')
    ap.add_argument('--out', default='output_p4/val_scores_full.pkl')
    args = ap.parse_args()

    feats = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))
    lip = pickle.load(open(BASE / 'output_lip/val_lip.pkl', 'rb'))
    print('lip inner keys sample:', list(lip['dev_seen'].keys())[:3])
    l0 = lip['dev_seen']['pair_000001']
    print('lip pair fields:', {k: (v.shape if hasattr(v, 'shape') else v) for k, v in l0.items()})

    model = MatcherV4().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True))
    model.eval()
    print(f'Loaded {args.model}')

    out = {}
    for split in ['dev_seen', 'dev_unseen']:
        preds = predict_split(model, feats[split], lip[split])
        y = [feats[split][p]['label'] for p in sorted(preds.keys())]
        pvals = [preds[p] for p in sorted(preds.keys())]
        auc = roc_auc_score(y, pvals)
        print(f'[{split}] AUC={auc:.6f} n={len(y)} pos={sum(y)}', flush=True)
        out[split] = preds
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
