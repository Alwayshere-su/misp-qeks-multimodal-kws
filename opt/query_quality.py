"""Query 质量评估模块

功能:
  - 对每个 query 计算:
    * effective frame count (有效帧数)
    * energy std (能量标准差)
    * silence ratio (静音占比)
    * embedding norm (嵌入范数)
    * max/mean similarity to word templates
    * similarity gap (max - mean)
  - 分类为 tiers: clean, noisy, truncated, unstable, very_low
  - 返回 quality dict per pid

用法:
  python opt/query_quality.py   # val 验证
"""

import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def assess_query_quality(data):
    """评估每个 query 的质量.

    Args:
        data: {pid: {query_frames, query_emb, enroll_emb, enroll_txt, ...}}

    Returns:
        dict[pid, dict]:
          - n_frames:         有效帧数
          - energy_std:       能量变异系数
          - silence_ratio:    静音帧占比
          - emb_norm:         嵌入 L2 范数
          - max_sim:          与同词模板的最大相似度
          - mean_sim:         与同词模板的均值相似度
          - sim_gap:          max_sim - mean_sim
          - quality_tier:     'clean'|'noisy'|'truncated'|'unstable'|'very_low'
    """
    # 按词分组
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))

    quality = {}

    for w, items in word_items.items():
        tn = np.stack([it['enroll_emb'] for _, it in items]).astype(np.float32)
        tn_n = tn / (np.linalg.norm(tn, axis=1, keepdims=True) + 1e-9)

        for pid, it in items:
            qf = it['query_frames'].astype(np.float32)
            qe = np.linalg.norm(qf, axis=1)

            # --- 有效帧 (非静音) ---
            energy_thresh = qe.max() * 0.05
            n_valid = int((qe > energy_thresh).sum())
            n_total = len(qe)

            # --- 能量稳定性 ---
            valid_qe = qe[qe > energy_thresh]
            if len(valid_qe) > 1:
                energy_std = float(np.std(valid_qe) / (valid_qe.mean() + 1e-9))
            else:
                energy_std = 1.0

            # --- 静音占比 ---
            silence_ratio = 1.0 - n_valid / max(n_total, 1)

            # --- 嵌入范数 ---
            emb_norm = float(np.linalg.norm(it['query_emb']))

            # --- 与同词模板的相似度 ---
            q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
            sims = q @ tn_n.T
            max_sim = float(sims.max())
            mean_sim = float(sims.mean())
            sim_gap = max_sim - mean_sim

            # --- 质量分级 ---
            if n_valid < 5 and silence_ratio > 0.7:
                tier = 'very_low'
            elif silence_ratio > 0.5 and sim_gap < 0.1:
                tier = 'unstable'
            elif n_valid < 8:
                tier = 'truncated'
            elif energy_std > 0.8 and sim_gap < 0.15:
                tier = 'noisy'
            else:
                tier = 'clean'

            quality[pid] = {
                'n_frames': n_valid,
                'energy_std': energy_std,
                'silence_ratio': silence_ratio,
                'emb_norm': emb_norm,
                'max_sim': max_sim,
                'mean_sim': mean_sim,
                'sim_gap': sim_gap,
                'quality_tier': tier,
            }

    return quality


def quality_summary(quality, pids):
    """打印质量分布摘要."""
    tier_counts = defaultdict(int)
    for pid in pids:
        tier_counts[quality[pid]['quality_tier']] += 1

    print("  Quality distribution:")
    for tier in ['clean', 'noisy', 'truncated', 'unstable', 'very_low']:
        c = tier_counts[tier]
        print(f"    {tier:12s}: {c:6d} ({c / len(pids) * 100:5.1f}%)")

    # 各 tier 的特征统计
    print("\n  Per-tier feature means:")
    for tier in ['clean', 'noisy', 'truncated', 'unstable', 'very_low']:
        tier_pids = [p for p in pids if quality[p]['quality_tier'] == tier]
        if not tier_pids:
            continue
        nf = np.mean([quality[p]['n_frames'] for p in tier_pids])
        sr = np.mean([quality[p]['silence_ratio'] for p in tier_pids])
        es = np.mean([quality[p]['energy_std'] for p in tier_pids])
        sg = np.mean([quality[p]['sim_gap'] for p in tier_pids])
        print(f"    {tier:12s}: n_frames={nf:5.1f}  silence={sr:.3f}  "
              f"energy_std={es:.3f}  sim_gap={sg:.4f}")


# ============================================================
# __main__: val 验证
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Query 质量评估 - Val 验证")
    print("=" * 60)

    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))

    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        pids = sorted(data.keys())
        print(f"\n--- {split} n={len(pids)} ---")

        quality = assess_query_quality(data)
        quality_summary(quality, pids)

        # 计算 per-tier AUC (如果有 label)
        if 'label' in data[pids[0]]:
            y = np.array([data[p]['label'] for p in pids])

            def roc_auc(yt, st):
                n_pos = int(yt.sum())
                n_neg = len(yt) - n_pos
                if n_pos == 0 or n_neg == 0:
                    return 0.5
                order = np.argsort(st)
                ranks = np.empty(len(st))
                ranks[order] = np.arange(1, len(st) + 1)
                return (ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

            print("\n  Per-tier max_sim AUC:")
            for tier in ['clean', 'noisy', 'truncated', 'unstable', 'very_low']:
                tier_pids = [p for p in pids if quality[p]['quality_tier'] == tier]
                if len(tier_pids) < 10:
                    continue
                yt = np.array([data[p]['label'] for p in tier_pids])
                st = np.array([quality[p]['max_sim'] for p in tier_pids])
                auc = roc_auc(yt, st)
                print(f"    {tier:12s}: n={len(tier_pids):5d}  AUC={auc:.4f}")

    print("\nDone!")
