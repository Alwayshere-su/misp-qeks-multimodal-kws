"""softmax 判别性多模板: query 与 W 模板的相似度相对全部词模板 (判别性)
score_W = softmax(s)[W], s_k = max sim(query, 词k模板)  (每词用质心代表)
"""
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def softmax_mt(data, T=1.0):
    """每词质心模板 -> query 与全部质心相似度 -> softmax 判别分数"""
    from collections import defaultdict
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))
    # 每词质心
    centroids = {}
    for w, items in word_items.items():
        c = np.mean(np.stack([it['enroll_emb'] for _, it in items]), axis=0)
        centroids[w] = c / (np.linalg.norm(c) + 1e-9)
    all_words = sorted(centroids.keys())
    C = np.stack([centroids[w] for w in all_words])  # (N_words, 384)
    scores = {}
    for pid, it in data.items():
        q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
        s = q @ C.T  # (N_words,)
        w = norm(it['enroll_txt'])
        wi = all_words.index(w)
        # softmax 判别
        sm = np.exp(s / T)
        sm = sm / sm.sum()
        scores[pid] = float(sm[wi])
    return scores


def main():
    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))
    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        y = [data[p]['label'] for p in sorted(data)]
        # 纯 max (基线)
        from collections import defaultdict
        word_items = defaultdict(list)
        for pid, it in data.items():
            word_items[norm(it['enroll_txt'])].append((pid, it))
        s_max = {}
        for w, items in word_items.items():
            T = np.stack([it['enroll_emb'] for _, it in items])
            Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
            for pid, it in items:
                q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
                s_max[pid] = float((q @ Tn.T).max())
        for T in [0.5, 1.0, 2.0]:
            s_soft = softmax_mt(data, T=T)
            auc = roc_auc_score(y, [s_soft[p] for p in sorted(data)])
            print(f'{split} softmax(T={T}): AUC={auc:.4f}', flush=True)
        print(f'{split} 纯max: AUC={roc_auc_score(y, [s_max[p] for p in sorted(data)]):.4f}', flush=True)


if __name__ == '__main__':
    main()
