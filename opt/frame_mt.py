"""帧级多模板匹配 (val 验证): query 帧 vs 同词所有 enroll 帧的对齐匹配"""
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def frame_mt(data):
    """帧级多模板: 对每词, query 帧与全部模板帧的 max-pool 相似度"""
    from collections import defaultdict
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))
    scores = {}
    for w, items in word_items.items():
        # 全部模板帧 (K*24, 384)
        T = np.concatenate([it['enroll_frames'].astype(np.float32) for _, it in items], axis=0)
        Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
        for pid, it in items:
            qf = it['query_frames'].astype(np.float32)
            qn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-9)
            sim = qn @ Tn.T  # (24, K*24)
            # max 池化: 每 query 帧的最近模板帧相似度平均
            s = float(sim.max(axis=1).mean())
            scores[pid] = s
    return scores


def main():
    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))
    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        fs = frame_mt(data)
        y = [data[p]['label'] for p in sorted(data)]
        s = np.array([fs[p] for p in sorted(data)])
        auc = roc_auc_score(y, s)
        print(f'{split}: 帧级多模板 AUC={auc:.4f}', flush=True)


if __name__ == '__main__':
    main()
