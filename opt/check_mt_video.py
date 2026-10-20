"""多模态多模板: 音频 + 视频模板匹配融合 (val 验证)"""
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def mt_scores(data, emb_key, qkey):
    from collections import defaultdict
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it.get('enroll_txt', it.get('enroll_txt', '')))].append((pid, it))
    scores = {}
    for w, items in word_items.items():
        templates = np.stack([it[emb_key] for _, it in items])
        tn = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-9)
        for pid, it in items:
            q = it[qkey] / (np.linalg.norm(it[qkey]) + 1e-9)
            sims = q @ tn.T
            scores[pid] = float(sims.max())
    return scores


def main():
    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))
    lip = pickle.load(open(BASE / 'output_lip/val_lip.pkl', 'rb'))
    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        ldata = lip[split]
        y = [data[p]['label'] for p in sorted(data)]
        sa = mt_scores(data, 'enroll_emb', 'query_emb')
        sv = mt_scores(ldata, 'enroll_video', 'query_video')
        s_audio = np.array([sa[p] for p in sorted(data)])
        s_video = np.array([sv[p] for p in sorted(data)])
        sa_r = rankdata(s_audio) / len(s_audio)
        sv_r = rankdata(s_video) / len(s_video)
        print(f'=== {split} ===', flush=True)
        print(f'  音频多模板 AUC={roc_auc_score(y, s_audio):.4f}', flush=True)
        print(f'  视频多模板 AUC={roc_auc_score(y, s_video):.4f}', flush=True)
        best = 0
        for w in np.arange(0, 1.01, 0.1):
            fused = w * sa_r + (1 - w) * sv_r
            auc = roc_auc_score(y, fused)
            if auc > best:
                best, bw = auc, w
        print(f'  融合 best w_audio={bw:.1f} AUC={best:.4f}', flush=True)


if __name__ == '__main__':
    main()
