"""s2u 声学打分: best_s2u.pt (dev_seen 训练) 对 dev_unseen 评估 — 词表外泛化 (接近 test_unseen)"""
import pickle
import sys
from pathlib import Path

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
    feats = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))
    lip = pickle.load(open(BASE / 'output_lip/val_lip.pkl', 'rb'))
    model = MatcherV4().to(DEVICE)
    model.load_state_dict(torch.load(BASE / 'output_p4/best_s2u.pt', map_location='cpu', weights_only=True))
    model.eval()
    print('Loaded best_s2u.pt (seen→unseen)', flush=True)
    out = {}
    for split in ['dev_seen', 'dev_unseen']:
        preds = predict_split(model, feats[split], lip[split])
        y = [feats[split][p]['label'] for p in sorted(preds.keys())]
        pvals = [preds[p] for p in sorted(preds.keys())]
        auc = roc_auc_score(y, pvals)
        print(f'[{split}] s2u AUC={auc:.6f} n={len(y)} pos={sum(y)}', flush=True)
        out[split] = preds
    with open(BASE / 'output_p4/s2u_scores.pkl', 'wb') as f:
        pickle.dump(out, f)
    print('saved output_p4/s2u_scores.pkl')


if __name__ == '__main__':
    main()
