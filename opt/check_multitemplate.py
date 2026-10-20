"""验证多模板匹配: query 与同词所有 enroll 音频的最大/平均相似度 (val 上有标签)"""
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
        # 按词分组
        from collections import defaultdict
        word_items = defaultdict(list)
        for pid, it in data.items():
            word_items[norm(it['enroll_txt'])].append((pid, it))
        print(f'=== {split}: {len(data)} pairs, {len(word_items)} 词 ===', flush=True)

        # 每个词: 收集所有 enroll_emb (模板) + 每个 pair 的 query
        y, s_single, s_multi_max, s_multi_mean = [], [], [], []
        for w, items in word_items.items():
            templates = np.stack([it['enroll_emb'] for _, it in items])  # (K,384)
            tn = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-9)
            for pid, it in items:
                q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
                # 单模板: 自己的 enroll
                e = it['enroll_emb'] / (np.linalg.norm(it['enroll_emb']) + 1e-9)
                s_single.append(float(q @ e))
                # 多模板
                sims = q @ tn.T
                s_multi_max.append(float(sims.max()))
                s_multi_mean.append(float(sims.mean()))
                y.append(it['label'])
        y = np.array(y)
        for name, s in [('单模板(自己enroll)', np.array(s_single)),
                        ('多模板max', np.array(s_multi_max)),
                        ('多模板mean', np.array(s_multi_mean))]:
            auc = roc_auc_score(y, s)
            print(f'  {name}: AUC={auc:.4f}', flush=True)
        # 多模板 + 文本? 简单融合
        print(flush=True)


if __name__ == '__main__':
    main()
