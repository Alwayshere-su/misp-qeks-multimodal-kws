"""稳健多模板聚合 - 多种评分变体

提供:
  - max (baseline)
  - top-k mean (top3, top5, top10, top10%)
  - robust = alpha * top3mean + (1-alpha) * max
  - norm_max = max - lambda * log(N_templates)
  - quality-weighted top-3 mean (模板质量 = norm_factor * consistency_with_centroid)
  - outlier template filtering (cross-similarity < mean-1.5*std 过滤)
  - softmax aggregation (configurable temperature)

数据结构: {pid: {enroll_emb: (384,), query_emb: (384,), enroll_txt: str, ...}}

用法:
  python opt/robust_mt.py   # val 验证
"""

import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    """归一化 enroll_txt: 小写+去标点"""
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def robust_mt_scores(data, alpha=0.6, lambda_n=0.02, softmax_temp=2.0):
    """稳健多模板聚合, 返回多种 scoring variant.

    Args:
        data: {pid: {enroll_emb, query_emb, enroll_txt, ...}}
        alpha: robust 组合系数 (alpha * top3mean + (1-alpha) * max)
        lambda_n: 模板数量校正系数
        softmax_temp: softmax 温度

    Returns:
        dict[str, dict[pid, score]]:
            'max', 'top3', 'top5', 'top10', 'top10pct',
            'robust', 'norm_max', 'quality', 'softmax'
    """
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))

    scores = {
        'max': {}, 'top3': {}, 'top5': {}, 'top10': {},
        'top10pct': {}, 'robust': {}, 'norm_max': {},
        'quality': {}, 'softmax': {},
    }

    for w, items in word_items.items():
        n = len(items)
        T = np.stack([it['enroll_emb'] for _, it in items]).astype(np.float64)
        tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)

        # ---- 单模板退化为 max ----
        if n == 1:
            for pid, it in items:
                q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
                v = float(q @ tn[0])
                for k in scores:
                    scores[k][pid] = v
            continue

        # ---- 模板质量权重 ----
        raw_n = np.linalg.norm(T, axis=1)
        centroid = tn.mean(0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        consist = tn @ centroid  # cosine to centroid
        nf = np.clip(raw_n / (raw_n.mean() + 1e-9), 0.3, 3.0)
        qw = np.clip(nf * np.clip(consist, 0.1, 2.0), 0.1, 10.0)

        # ---- 离群模板检测 ----
        cross = tn @ tn.T
        np.fill_diagonal(cross, 0)
        acs = cross.mean(1)
        thresh = max(acs.mean() - 1.5 * acs.std(), 0.3)
        good = acs >= thresh
        gtn, gw = tn[good], qw[good]
        ng = max(len(gtn), 1)

        # ---- 逐 query 计算 ----
        for pid, it in items:
            q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
            sims = q @ tn.T
            ss = np.sort(sims)[::-1]

            # --- 基础变体 ---
            scores['max'][pid] = float(sims.max())

            for k in [3, 5, 10]:
                ke = min(k, n)
                scores[f'top{k}'][pid] = float(ss[:ke].mean())

            kp = max(1, int(n * 0.1))
            scores['top10pct'][pid] = float(ss[:kp].mean())

            # --- 进阶变体 ---
            k3 = min(3, n)
            scores['robust'][pid] = (
                alpha * float(ss[:k3].mean()) + (1 - alpha) * scores['max'][pid]
            )

            scores['norm_max'][pid] = scores['max'][pid] - lambda_n * np.log(n)

            # quality-weighted top-3 (只用 good 模板)
            gs = q @ gtn.T
            gi = np.argsort(gs)[::-1]
            kq = min(3, ng)
            scores['quality'][pid] = float(
                np.average(gs[gi[:kq]], weights=gw[gi[:kq]])
            )

            # softmax aggregation
            sm = np.exp(sims / softmax_temp)
            sm /= sm.sum()
            scores['softmax'][pid] = float(sm.max())

    return scores


def roc_auc(y, s):
    """手动计算 AUC (不依赖 sklearn)"""
    y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ============================================================
# __main__: val 验证
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("稳健多模板聚合 - Val 验证")
    print("=" * 60)

    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))

    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        pids = sorted(data.keys())
        y = np.array([data[p]['label'] for p in pids])
        print(f"\n--- {split} n={len(pids)} pos={y.mean():.3f} ---")

        # 默认参数
        mt = robust_mt_scores(data, alpha=0.6, lambda_n=0.02, softmax_temp=2.0)
        best_auc, best_name = 0.0, ''

        variant_names = [
            'max', 'top3', 'top5', 'top10', 'top10pct',
            'robust', 'norm_max', 'quality', 'softmax',
        ]
        for k in variant_names:
            s = np.array([mt[k][p] for p in pids])
            auc = roc_auc(y, s)
            marker = ' ***' if auc > best_auc + 0.0001 else ''
            if auc > best_auc:
                best_auc, best_name = auc, k
            print(f"  {k:12s}: AUC={auc:.4f}{marker}")

        bl = roc_auc(y, np.array([mt['max'][p] for p in pids]))
        print(f"  Baseline max AUC={bl:.4f}, "
              f"Best={best_name} AUC={best_auc:.4f} (+{best_auc - bl:.4f})")

    print("\nDone!")
