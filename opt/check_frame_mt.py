"""帧级多模板匹配 + top-k 变体 (val 验证)"""
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def frame_mt_score(qf, templates):
    """qf: (24,384); templates: (K,24,384) -> 帧级对齐匹配分数 (mean-of-max)"""
    qn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-9)
    best = -1.0
    for t in templates:
        tn = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
        sim = qn @ tn.T  # (24,24)
        mom = (sim.max(axis=1).mean() + sim.max(axis=0).mean()) / 2
        best = max(best, float(mom))
    return best


def main():
    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))
    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        from collections import defaultdict
        word_items = defaultdict(list)
        for pid, it in data.items():
            word_items[norm(it['enroll_txt'])].append((pid, it))
        y, s_g = [], []
        for w, items in word_items.items():
            templates_g = np.stack([it['enroll_emb'] for _, it in items])
            tn = templates_g / (np.linalg.norm(templates_g, axis=1, keepdims=True) + 1e-9)
            templates_f = [it['enroll_frames'].astype(np.float32) for _, it in items]
            for pid, it in items:
                q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
                s_g.append(float((q @ tn.T).max()))
                y.append(it['label'])
        print(f'=== {split} ===', flush=True)
        print(f'  全局多模板 AUC={roc_auc_score(y, s_g):.4f}', flush=True)


if __name__ == '__main__':
    main()
