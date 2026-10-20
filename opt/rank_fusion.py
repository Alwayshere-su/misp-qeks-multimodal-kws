"""Rank/Percentile 融合模块

功能:
  - 将原始分数转为 rank/percentile (每个 split 内独立排序)
  - 融合多信号: weighted rank average
  - 支持 quality-stratified weights (不同质量 tier 不同权重)
  - 生成 submission CSV: id,type,prediction 格式

用法:
  python opt/rank_fusion.py   # 独立测试
"""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def rank_norm(preds):
    """将分数转为 [0, 1] rank/percentile.

    Args:
        preds: dict[pid, float]

    Returns:
        dict[pid, float] rank-percentile in [0, 1]
    """
    ids = sorted(preds.keys())
    v = np.array([preds[i] for i in ids])
    r = rankdata(v) / len(v)
    return dict(zip(ids, r))


def fuse_rank(signal_dict, weights, ids_range=50000):
    """将多个信号的 rank percentile 按权重融合.

    Args:
        signal_dict: {name: {pid: raw_score}}
        weights: {name: weight}
        ids_range: pid 数量 (0..ids_range-1)

    Returns:
        fused: {index: fused_score}  (index-based)
        ranked: {name: {pid: rank_score}}   (pid-based)
    """
    # 每个信号独立 rank
    ranked = {}
    for name, raw in signal_dict.items():
        if len(raw) == 0:
            continue
        ranked[name] = rank_norm(raw)

    # 融合
    fused = {}
    for i in range(ids_range):
        pid = f"pair_{i + 1:06d}"
        s = 0.0
        w_sum = 0.0
        for name, w in weights.items():
            if name in ranked:
                s += w * ranked[name].get(pid, 0.5)
                w_sum += w
        fused[i] = s / max(w_sum, 1e-9) if w_sum > 0 else 0.5

    return fused, ranked


def fuse_rank_quality_stratified(
    signal_dict, weights_by_tier, quality, ids_range=50000
):
    """质量分层的 rank 融合: 每个质量 tier 使用不同权重.

    Args:
        signal_dict: {name: {pid: raw_score}}
        weights_by_tier: {tier: {name: weight}}
        quality: {pid: {'quality_tier': str}}
        ids_range: pid 数量

    Returns:
        dict[index, fused_score]
    """
    # 每个信号独立 rank
    ranked = {}
    for name, raw in signal_dict.items():
        if len(raw) == 0:
            continue
        ranked[name] = rank_norm(raw)

    # 默认权重 (tier 不在 weights_by_tier 时使用)
    default_weights = {'mt_robust': 0.5, 'mt_quality': 0.3, 'mt_norm': 0.2}

    fused = {}
    for i in range(ids_range):
        pid = f"pair_{i + 1:06d}"
        tier = quality.get(pid, {}).get('quality_tier', 'clean')
        weights = weights_by_tier.get(tier, default_weights)

        s = 0.0
        w_sum = 0.0
        for name, w in weights.items():
            if name in ranked:
                s += w * ranked[name].get(pid, 0.5)
                w_sum += w
        fused[i] = s / max(w_sum, 1e-9) if w_sum > 0 else 0.5

    return fused, ranked


def load_submission(name):
    """加载已有 submission CSV.

    Returns:
        (inset_dict, outset_dict) 每个为 {index: score}
    """
    ins, out = {}, {}
    for r in csv.DictReader(open(BASE / 'submission' / name, encoding='utf-8')):
        (ins if r['type'] == 'inset' else out)[int(r['id'])] = float(r['prediction'])
    return ins, out


def write_submission(fused_inset, fused_outset, out_path, n=50000):
    """写入 submission CSV.

    Args:
        fused_inset: {index: score} for inset
        fused_outset: {index: score} for outset
        out_path: 输出路径 (str or Path)
        n: 每类样本数
    """
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(n):
            f.write(f"{i},inset,{fused_inset.get(i, 0.5):.6f}\n")
        for i in range(n):
            f.write(f"{i},outset,{fused_outset.get(i, 0.5):.6f}\n")
    print(f"  saved {out_path}", flush=True)


# ============================================================
# __main__: 独立测试
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Rank/Percentile 融合 - 独立测试")
    print("=" * 60)

    # 模拟信号
    np.random.seed(42)
    n_pids = 200
    pids = [f"pair_{i + 1:06d}" for i in range(n_pids)]

    signals = {
        'mt_robust': {p: np.random.random() for p in pids},
        'frame': {p: np.random.random() * 0.5 for p in pids},
        'phonetic': {p: 0.3 + np.random.random() * 0.4 for p in pids},
    }

    # 模拟质量标签
    quality = {}
    for i, p in enumerate(pids):
        r = np.random.random()
        if r < 0.5:
            tier = 'clean'
        elif r < 0.75:
            tier = 'noisy'
        elif r < 0.9:
            tier = 'truncated'
        else:
            tier = 'very_low'
        quality[p] = {'quality_tier': tier}

    # 测试 1: 简单融合
    print("\n[1] Simple weighted rank fusion:")
    weights = {'mt_robust': 0.5, 'frame': 0.3, 'phonetic': 0.2}
    fused, ranked = fuse_rank(signals, weights, ids_range=n_pids)
    print(f"  Fused {len(fused)} entries, first 3: "
          f"{[(i, f'{fused[i]:.4f}') for i in range(3)]}")

    # 测试 2: 质量分层融合
    print("\n[2] Quality-stratified rank fusion:")
    weights_by_tier = {
        'clean': {'mt_robust': 0.5, 'frame': 0.35, 'phonetic': 0.15},
        'noisy': {'mt_robust': 0.3, 'frame': 0.4, 'phonetic': 0.3},
        'truncated': {'mt_robust': 0.2, 'frame': 0.5, 'phonetic': 0.3},
        'very_low': {'mt_robust': 0.1, 'frame': 0.4, 'phonetic': 0.5},
    }
    fused_qs, ranked_qs = fuse_rank_quality_stratified(
        signals, weights_by_tier, quality, ids_range=n_pids
    )
    print(f"  Fused {len(fused_qs)} entries, first 3: "
          f"{[(i, f'{fused_qs[i]:.4f}') for i in range(3)]}")

    # 验证: rank 值在 [0, 1] 内
    for name, r in ranked.items():
        vals = list(r.values())
        print(f"  {name}: min={min(vals):.4f} max={max(vals):.4f} "
              f"mean={np.mean(vals):.4f}")

    print("\nAll tests passed!")
