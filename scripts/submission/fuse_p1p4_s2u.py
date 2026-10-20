"""p1 s2u 模型对 val 打分 + 与 p4 s2u 融合评估 (dev_unseen 无泄漏)"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_whisper_p1 import MatcherV1, encode_text

BASE = Path(r"D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
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


def main():
    feats = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))
    p4s2u = pickle.load(open(BASE / 'output_p4/s2u_scores.pkl', 'rb'))

    model = MatcherV1().to(DEVICE)
    model.load_state_dict(torch.load(BASE / 'output_p1/best_s2u.pt', map_location='cpu', weights_only=True))
    model.eval()
    print('Loaded p1 best_s2u.pt', flush=True)

    for split in ['dev_seen', 'dev_unseen']:
        preds = predict_split(model, feats[split])
        y = [feats[split][p]['label'] for p in sorted(preds.keys())]
        p1 = np.array([preds[p] for p in sorted(preds.keys())])
        p4 = np.array([p4s2u[split][p] for p in sorted(preds.keys())])
        y = np.array(y)
        print(f'=== {split} n={len(y)} pos={y.sum()} ===', flush=True)
        print(f'  p1 alone:  AUC={roc_auc_score(y, p1):.6f}', flush=True)
        print(f'  p4 alone:  AUC={roc_auc_score(y, p4):.6f}', flush=True)
        p1r = rankdata(p1) / len(p1)
        p4r = rankdata(p4) / len(p4)
        best_w, best_auc = 0.0, 0.0
        for w in np.arange(0.0, 1.01, 0.1):
            fused = w * p1r + (1 - w) * p4r
            auc = roc_auc_score(y, fused)
            if auc > best_auc:
                best_auc, best_w = auc, w
        print(f'  BEST w_p1={best_w:.1f} AUC={best_auc:.6f}', flush=True)
        # 保存 p1 分数供后续用
        with open(BASE / f'output_p1/p1_{split}_s2u.pkl', 'wb') as f:
            pickle.dump(preds, f)


if __name__ == '__main__':
    main()
