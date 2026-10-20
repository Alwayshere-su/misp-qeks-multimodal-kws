# 验证关键词原型检索 (val 留一法) — 不用 train 数据
import pickle, numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict

d = pickle.load(open(r'output_whisper/val_feats.pkl', 'rb'))
for name in ['dev_seen', 'dev_unseen']:
    items = list(d[name].values())
    groups = defaultdict(list)
    for pid, it in d[name].items():
        groups[it['enroll_txt']].append((pid, it['enroll_emb'], it['query_emb'], it['label']))
    y, s = [], []
    for kw, g in groups.items():
        if len(g) < 2:
            for pid, e, q, lab in g:
                e = e / np.linalg.norm(e); q = q / np.linalg.norm(q)
                y.append(lab); s.append(float(e @ q))
            continue
        embs = np.stack([e for _, e, _, _ in g])
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        for i, (pid, e, q, lab) in enumerate(g):
            proto = np.mean(np.delete(embs, i, axis=0), axis=0)
            proto = proto / np.linalg.norm(proto)
            q = q / np.linalg.norm(q)
            y.append(lab); s.append(float(q @ proto))
    print(f'{name}: kw={len(groups)}, n={len(y)}, proto_AUC={roc_auc_score(y, s):.4f}')
