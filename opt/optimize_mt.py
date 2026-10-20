"""多模板变体优化 (val 无泄漏验证): max / topk / 对比归一化"""
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
        y, s_max, s_top2, s_top3, s_norm = [], [], [], [], []
        # 全部模板嵌入 (用于对比归一化)
        all_templates = []
        for w, items in word_items.items():
            t = np.stack([it['enroll_emb'] for _, it in items])
            tn = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
            all_templates.append(tn)
        all_T = np.concatenate(all_templates, axis=0)  # (全部模板, 384)
        all_Tn = all_T / (np.linalg.norm(all_T, axis=1, keepdims=True) + 1e-9)
        for w, items in word_items.items():
            templates = np.stack([it['enroll_emb'] for _, it in items])
            tn = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-9)
            for pid, it in items:
                q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
                sims = q @ tn.T
                s_max.append(float(sims.max()))
                s_top2.append(float(np.sort(sims)[-2:].mean()))
                s_top3.append(float(np.sort(sims)[-3:].mean()))
                # 对比归一化: W模板最大相似 - 全模板平均相似
                all_sims = q @ all_Tn.T
                s_norm.append(float(sims.max() - all_sims.mean()))
                y.append(it['label'])
        print(f'=== {split} ===', flush=True)
        for name, s in [('max', s_max), ('top2', s_top2), ('top3', s_top3),
                        ('max-全模板均值', s_norm)]:
            print(f'  {name}: AUC={roc_auc_score(y, s):.4f}', flush=True)
        # max-全模板均值 与 max 融合
        a = np.array(s_max); b = np.array(s_norm)
        from scipy.stats import rankdata
        ar, br = rankdata(a) / len(a), rankdata(b) / len(b)
        best = 0
        for w in np.arange(0, 1.01, 0.1):
            auc = roc_auc_score(y, w * ar + (1 - w) * br)
            if auc > best:
                best, bw = auc, w
        print(f'  max + 对比归一化 融合 best w_max={bw:.1f} AUC={best:.4f}', flush=True)


if __name__ == '__main__':
    main()
