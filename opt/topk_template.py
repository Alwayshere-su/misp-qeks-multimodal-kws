"""多模板改进验证: 模板侧 top-k 平均 (val)"""
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def main():
    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))
    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        from collections import defaultdict
        word_items = defaultdict(list)
        for pid, it in data.items():
            word_items[norm(it['enroll_txt'])].append((pid, it))
        y, s_max, s_t2, s_t3 = [], [], [], []
        for w, items in word_items.items():
            T = np.stack([it['enroll_emb'] for _, it in items])
            Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
            for pid, it in items:
                q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
                sims = q @ Tn.T
                s_max.append(float(sims.max()))
                s_t2.append(float(np.sort(sims)[-2:].mean()))
                s_t3.append(float(np.sort(sims)[-3:].mean()))
                y.append(it['label'])
        print(f'=== {split} ===', flush=True)
        for name, s in [('模板max', s_max), ('模板top2平均', s_t2), ('模板top3平均', s_t3)]:
            print(f'  {name}: AUC={roc_auc_score(y, s):.4f}', flush=True)


if __name__ == '__main__':
    main()
