"""帧级 DTW 匹配模块

功能:
  - 提取帧嵌入 (enroll_frames (24,384), query_frames (24,384))
  - 去除静音帧 (energy < 5% of max)
  - 计算 cosine 距离矩阵
  - Constrained DTW with Sakoe-Chiba window (window = max(5, 0.5*max(Q,E)))
  - 返回归一化 DTW score per pair
  - fast frame multi-template variant (max-pool on frame similarities, top-50% best)

用法:
  python opt/frame_dtw.py   # val 验证 (抽样)
"""

import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get('MISP_ROOT', Path(__file__).resolve().parents[1]))


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def constrained_dtw(dist_matrix, window):
    """带 Sakoe-Chiba 窗口约束的 DTW (返回归一化累积距离).

    Args:
        dist_matrix: (Q, E) cosine distance matrix
        window: Sakoe-Chiba band width

    Returns:
        normalized cumulative distance (越小越好)
    """
    Q, E = dist_matrix.shape
    D = np.full((Q + 1, E + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0

    for i in range(1, Q + 1):
        j_start = max(1, i - window)
        j_end = min(E + 1, i + window + 1)
        for j in range(j_start, j_end):
            cost = dist_matrix[i - 1, j - 1]
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    path_len = Q + E
    return D[Q, E] / max(path_len, 1)


def remove_silence_frames(frames, energy_thresh_ratio=0.05):
    """去除低能量(静音)帧.

    Args:
        frames: (N, D) frame embeddings
        energy_thresh_ratio: 能量阈值比例 (相对于 max)

    Returns:
        (M, D) cleaned frames, bool mask
    """
    energy = np.linalg.norm(frames, axis=1)
    thresh = energy.max() * energy_thresh_ratio
    valid = energy > thresh
    if valid.sum() < 2:
        valid = np.ones(len(frames), dtype=bool)
    return frames[valid], valid


def frame_dtw_scores(data):
    """逐对 DTW 匹配分数.

    对每个 (query, enroll) 对:
      1. 去除双方静音帧
      2. 归一化帧嵌入
      3. 计算 cosine 距离矩阵
      4. 约束 DTW
      5. 返回 -dtw_score 作为相似度 (越大越好)

    Returns:
        dict[pid, float] 相似度分数
    """
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))

    scores = {}

    for w, items in word_items.items():
        for pid, it in items:
            qf = it['query_frames'].astype(np.float32)
            qf_clean, _ = remove_silence_frames(qf)
            qf_clean = qf_clean / (np.linalg.norm(qf_clean, axis=1, keepdims=True) + 1e-9)

            # 对所有 enroll 模板做 DTW, 取最佳
            dtw_vals = []
            for _, it_t in items:
                ef = it_t['enroll_frames'].astype(np.float32)
                ef_clean, _ = remove_silence_frames(ef)
                ef_clean = ef_clean / (np.linalg.norm(ef_clean, axis=1, keepdims=True) + 1e-9)

                # cosine distance = 1 - cosine similarity
                dist = 1.0 - qf_clean @ ef_clean.T

                w_size = max(5, int(0.5 * max(len(qf_clean), len(ef_clean))))
                dtw_val = constrained_dtw(dist, w_size)
                dtw_vals.append(dtw_val)

            # 取最小 DTW 距离 (最匹配), 翻转为相似度
            best_dtw = float(np.min(dtw_vals))
            scores[pid] = -best_dtw  # 负距离 = 相似度

    return scores


def fast_frame_mt(data):
    """快速帧级多模板: max-pool + top-50% 均值.

    对每词:
      - 拼接所有 enroll 模板帧 -> (K*24, 384)
      - query 帧与全部模板帧做 cosine similarity
      - 每 query 帧取最佳模板帧 -> per-frame max
      - 取 top-50% 帧的均值作为分数

    Returns:
        dict[pid, float]
    """
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))

    scores = {}

    for w, items in word_items.items():
        T = np.concatenate([it['enroll_frames'].astype(np.float32) for _, it in items], axis=0)
        Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)

        for pid, it in items:
            qf = it['query_frames'].astype(np.float32)
            qn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-9)

            # per-frame best match
            sim = qn @ Tn.T  # (Q, K*24)
            per_frame_max = sim.max(axis=1)

            # top-50% 帧均值
            k = max(1, len(per_frame_max) // 2)
            scores[pid] = float(np.sort(per_frame_max)[-k:].mean())

    return scores


def roc_auc(y, s):
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
    import random

    print("=" * 60)
    print("帧级 DTW 匹配 - Val 验证")
    print("=" * 60)

    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))

    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        pids = sorted(data.keys())
        y_all = np.array([data[p]['label'] for p in pids])

        # 抽样加速 DTW 验证
        random.seed(42)
        n_sample = min(2000, len(pids))
        sample_pids = random.sample(pids, n_sample)
        sample_data = {p: data[p] for p in sample_pids}
        y_sample = np.array([data[p]['label'] for p in sample_pids])

        print(f"\n--- {split} (sampled {n_sample}/{len(pids)}) ---")

        # --- 快速帧级多模板 ---
        print("  Computing fast_frame_mt...", end=' ', flush=True)
        fm = fast_frame_mt(data)
        s_fm = np.array([fm[p] for p in pids])
        auc_fm = roc_auc(y_all, s_fm)
        print(f"AUC={auc_fm:.4f}")

        # --- 逐对 DTW (抽样) ---
        print("  Computing frame_dtw (sampled)...", end=' ', flush=True)
        dtw = frame_dtw_scores(sample_data)
        s_dtw = np.array([dtw[p] for p in sample_pids])
        auc_dtw = roc_auc(y_sample, s_dtw)
        print(f"AUC={auc_dtw:.4f}")

        print(f"  fast_frame_mt AUC={auc_fm:.4f}  |  dtw AUC={auc_dtw:.4f} (sampled)")

    print("\nDone!")
