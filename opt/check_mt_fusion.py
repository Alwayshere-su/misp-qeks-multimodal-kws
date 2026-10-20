"""val 融合验证: 多模板 + p4 模型预测 (确定融合权重)"""
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def mt_scores(data):
    from collections import defaultdict
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))
    scores = {}
    for w, items in word_items.items():
        templates = np.stack([it['enroll_emb'] for _, it in items])
        tn = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-9)
        for pid, it in items:
            q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
            scores[pid] = float((q @ tn.T).max())
    return scores


def main():
    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))
    s2u = pickle.load(open(BASE / 'output_p4/s2u_scores.pkl', 'rb'))  # p4 seen→unseen 模型
    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        y = np.array([data[p]['label'] for p in sorted(data)])
        ms = mt_scores(data)
        mt = np.array([ms[p] for p in sorted(data)])
        p4 = np.array([s2u[split].get(p, 0.5) for p in sorted(data)])
        mt_r = rankdata(mt) / len(mt)
        p4_r = rankdata(p4) / len(p4)
        print(f'=== {split} ===', flush=True)
        print(f'  多模板 AUC={roc_auc_score(y, mt):.4f}  p4 AUC={roc_auc_score(y, p4):.4f}', flush=True)
        best = 0
        for w in np.arange(0, 1.01, 0.1):
            auc = roc_auc_score(y, w * mt_r + (1 - w) * p4_r)
            if auc > best:
                best, bw = auc, w
        print(f'  融合 best w_mt={bw:.1f} AUC={best:.4f}', flush=True)


if __name__ == '__main__':
    main()
