"""健壮多模板主管线 - 整合全部 5 大改进

改进点:
  1. 稳健多模板聚合 (robust_mt.py)
  2. 帧级 DTW + 快速帧级多模板 (frame_dtw.py)
  3. enrollment 文本发音先验 (phoneme_prior.py)
  4. query 质量检测 + 分层评分 (query_quality.py)
  5. rank/percentile 融合 (rank_fusion.py)

输出:
  - submit_robust_mt.csv:            纯稳健多模板 rank 融合
  - submit_robust_mt_p4_v1..v4.csv:  多模板 + p4 融合 (不同权重)
  - submit_robust_mt_quality_split.csv: 质量分层融合
  - submit_mt_robust_only.csv 等:    各子信号独立提交

用法:
  python opt/robust_pipeline.py   # 直接生成 test 提交
"""

import csv
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')

# 导入子模块
sys.path.insert(0, str(BASE / 'opt'))
from robust_mt import robust_mt_scores as _robust_mt_scores
from frame_dtw import fast_frame_mt as _fast_frame_mt
from frame_dtw import frame_dtw_scores as _frame_dtw_scores
from phoneme_prior import compute_phoneme_scores as _compute_phoneme_scores
from query_quality import assess_query_quality as _assess_query_quality
from rank_fusion import (
    rank_norm, fuse_rank, fuse_rank_quality_stratified,
    load_submission, write_submission,
)


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


# ============================================================
# 主管线: Test 提交生成
# ============================================================

def run_submit():
    """生成全部 test 提交变体."""
    print("=" * 60)
    print("健壮多模板主管线 - Test 提交生成")
    print("=" * 60)

    # ---- 加载 test 数据 ----
    print("\n[加载] test_inset.pkl + test_outset.pkl ...", flush=True)
    tin = pickle.load(open(BASE / 'output_whisper/test_inset.pkl', 'rb'))
    tout = pickle.load(open(BASE / 'output_whisper/test_outset.pkl', 'rb'))
    print(f"  inset: {len(tin)} pairs, outset: {len(tout)} pairs", flush=True)

    # ---- 1. 稳健多模板聚合 ----
    print("\n[1/5] 稳健多模板聚合 (global embedding variants)...", flush=True)
    mt_i = _robust_mt_scores(tin, alpha=0.6, lambda_n=0.02, softmax_temp=2.0)
    mt_o = _robust_mt_scores(tout, alpha=0.6, lambda_n=0.02, softmax_temp=2.0)
    for k in ['max', 'robust', 'norm_max', 'quality', 'softmax']:
        print(f"  mt_{k}: inset {len(mt_i[k])}  outset {len(mt_o[k])}", flush=True)

    # ---- 2. 帧级多模板 ----
    print("\n[2/5] 帧级多模板 (fast frame MT)...", flush=True)
    frame_i = _fast_frame_mt(tin)
    frame_o = _fast_frame_mt(tout)
    print(f"  frame_mt: inset {len(frame_i)}  outset {len(frame_o)}", flush=True)

    # ---- 3. 发音先验 ----
    print("\n[3/5] 发音先验 (phoneme prior)...", flush=True)
    ph_i = _compute_phoneme_scores(tin)
    ph_o = _compute_phoneme_scores(tout)
    print(f"  phoneme: inset {len(ph_i)}  outset {len(ph_o)}", flush=True)

    # ---- 4. query 质量检测 ----
    print("\n[4/5] query 质量检测...", flush=True)
    quality_i = _assess_query_quality(tin)
    quality_o = _assess_query_quality(tout)
    for label, q in [('inset', quality_i), ('outset', quality_o)]:
        tc = defaultdict(int)
        for v in q.values():
            tc[v['quality_tier']] += 1
        print(f"  {label}: {dict(sorted(tc.items()))}", flush=True)

    # ---- 5. Rank 融合 + 提交生成 ----
    print("\n[5/5] Rank 融合 + 提交生成...", flush=True)

    # 构建信号字典
    def build_signals(data, mt, frame, ph):
        pids = sorted(data.keys())
        return {
            'mt_robust':  {p: mt['robust'].get(p, mt['max'].get(p, 0.5)) for p in pids},
            'mt_norm':    {p: mt['norm_max'].get(p, mt['max'].get(p, 0.5)) for p in pids},
            'mt_quality': {p: mt['quality'].get(p, mt['max'].get(p, 0.5)) for p in pids},
            'mt_softmax': {p: mt['softmax'].get(p, mt['max'].get(p, 0.5)) for p in pids},
            'frame':      {p: frame.get(p, 0.0) for p in pids},
            'phonetic':   {p: ph.get(p, 0.5) for p in pids},
        }

    sig_i = build_signals(tin, mt_i, frame_i, ph_i)
    sig_o = build_signals(tout, mt_o, frame_o, ph_o)

    # ---- 融合权重 ----
    # inset: 全局模板更可靠 (训练分布内)
    w_inset = {
        'mt_robust': 0.45, 'mt_quality': 0.15,
        'frame': 0.15, 'phonetic': 0.05, 'mt_norm': 0.15, 'mt_softmax': 0.05,
    }
    # outset: 帧级+音素权重更高 (分布外)
    w_outset = {
        'mt_robust': 0.40, 'mt_quality': 0.10,
        'frame': 0.20, 'phonetic': 0.10, 'mt_norm': 0.15, 'mt_softmax': 0.05,
    }

    fused_i, ranked_i = fuse_rank(sig_i, w_inset, 50000)
    fused_o, ranked_o = fuse_rank(sig_o, w_outset, 50000)

    # ---- a) 纯稳健多模板提交 ----
    print("\n  [a] submit_robust_mt.csv (pure robust MT)...", flush=True)
    write_submission(fused_i, fused_o, BASE / 'submission/submit_robust_mt.csv')

    # ---- b) 各子信号独立提交 ----
    print("\n  [b] Individual signal submissions...", flush=True)
    sig_list = ['mt_robust', 'mt_norm', 'mt_quality', 'mt_softmax', 'frame', 'phonetic']
    for sig_name in sig_list:
        ri = rank_norm(sig_i[sig_name])
        ro = rank_norm(sig_o[sig_name])
        out_path = BASE / f'submission/submit_{sig_name}_only.csv'
        write_submission(ri, ro, out_path)

    # ---- c) 与 p4_full 融合 ----
    print("\n  [c] Robust MT + p4_full fusion variants...", flush=True)
    full = load_submission('submit_p4_full.csv')
    rf_i, rf_o = rank_norm(full[0]), rank_norm(full[1])

    variants = {
        'v1': (0.50, 0.60),
        'v2': (0.55, 0.65),
        'v3': (0.45, 0.55),
        'v4': (0.60, 0.70),
    }

    for tag, (w_i, w_o) in variants.items():
        with open(BASE / f'submission/submit_robust_mt_p4_{tag}.csv',
                  'w', encoding='utf-8', newline='') as f:
            f.write("id,type,prediction\n")
            for i in range(50000):
                sc = w_i * fused_i.get(i, 0.5) + (1 - w_i) * rf_i.get(i, 0.5)
                f.write(f"{i},inset,{sc:.6f}\n")
            for i in range(50000):
                sc = w_o * fused_o.get(i, 0.5) + (1 - w_o) * rf_o.get(i, 0.5)
                f.write(f"{i},outset,{sc:.6f}\n")
        print(f"    saved submit_robust_mt_p4_{tag}.csv (w_inset={w_i}, w_outset={w_o})",
              flush=True)

    # ---- d) 质量分层融合 ----
    print("\n  [d] Quality-stratified fusion...", flush=True)

    # 质量分层权重: 低质量样本更依赖多模板 (不依赖 p4)
    tier_weights = {
        'clean':     {'mt': 0.50,  'p4': 0.50},
        'noisy':     {'mt': 0.60,  'p4': 0.40},
        'unstable':  {'mt': 0.65,  'p4': 0.35},
        'truncated': {'mt': 0.70,  'p4': 0.30},
        'very_low':  {'mt': 0.80,  'p4': 0.20},
    }

    with open(BASE / 'submission/submit_robust_mt_quality_split.csv',
              'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            pid = f"pair_{i + 1:06d}"
            tier = quality_i.get(pid, {}).get('quality_tier', 'clean')
            tw = tier_weights.get(tier, {'mt': 0.5, 'p4': 0.5})
            sc = tw['mt'] * fused_i.get(i, 0.5) + tw['p4'] * rf_i.get(i, 0.5)
            f.write(f"{i},inset,{sc:.6f}\n")
        for i in range(50000):
            pid = f"pair_{50000 + i + 1:06d}"
            tier = quality_o.get(pid, {}).get('quality_tier', 'clean')
            tw = tier_weights.get(tier, {'mt': 0.5, 'p4': 0.5})
            sc = tw['mt'] * fused_o.get(i, 0.5) + tw['p4'] * rf_o.get(i, 0.5)
            f.write(f"{i},outset,{sc:.6f}\n")
    print("    saved submit_robust_mt_quality_split.csv", flush=True)

    # ---- e) 额外变体: 不同 alpha 的 robust MT + p4 ----
    print("\n  [e] Additional alpha variants...", flush=True)
    for alpha_val in [0.5, 0.7]:
        mt_i_a = _robust_mt_scores(tin, alpha=alpha_val)
        mt_o_a = _robust_mt_scores(tout, alpha=alpha_val)
        sig_i_a = build_signals(tin, mt_i_a, frame_i, ph_i)
        sig_o_a = build_signals(tout, mt_o_a, frame_o, ph_o)
        fused_i_a, _ = fuse_rank(sig_i_a, w_inset, 50000)
        fused_o_a, _ = fuse_rank(sig_o_a, w_outset, 50000)

        with open(BASE / f'submission/submit_robust_mt_alpha{alpha_val}_p4.csv',
                  'w', encoding='utf-8', newline='') as f:
            f.write("id,type,prediction\n")
            w_i, w_o = 0.55, 0.65
            for i in range(50000):
                sc = w_i * fused_i_a.get(i, 0.5) + (1 - w_i) * rf_i.get(i, 0.5)
                f.write(f"{i},inset,{sc:.6f}\n")
            for i in range(50000):
                sc = w_o * fused_o_a.get(i, 0.5) + (1 - w_o) * rf_o.get(i, 0.5)
                f.write(f"{i},outset,{sc:.6f}\n")
        print(f"    saved submit_robust_mt_alpha{alpha_val}_p4.csv", flush=True)

    # ---- 总结 ----
    print("\n" + "=" * 60)
    print("全部提交已生成!")
    print("=" * 60)
    print("\n生成的文件列表:")
    submitted = [
        'submit_robust_mt.csv',
        'submit_robust_mt_p4_v1.csv',
        'submit_robust_mt_p4_v2.csv',
        'submit_robust_mt_p4_v3.csv',
        'submit_robust_mt_p4_v4.csv',
        'submit_robust_mt_quality_split.csv',
        'submit_robust_mt_alpha0.5_p4.csv',
        'submit_robust_mt_alpha0.7_p4.csv',
    ]
    for sig_name in sig_list:
        submitted.append(f'submit_{sig_name}_only.csv')
    for s in submitted:
        print(f"  submission/{s}")


# ============================================================
# Val 快速验证 (抽样)
# ============================================================

def run_val():
    """快速 val 验证 (各子模块的 AUC)."""
    print("=" * 60)
    print("健壮多模板主管线 - Val 快速验证")
    print("=" * 60)

    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))

    def roc_auc(y, s):
        y, s = np.asarray(y, float), np.asarray(s, float)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        order = np.argsort(s)
        ranks = np.empty(len(s))
        ranks[order] = np.arange(1, len(s) + 1)
        return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        pids = sorted(data.keys())
        y = np.array([data[p]['label'] for p in pids])
        print(f"\n--- {split} n={len(pids)} ---")

        # 各信号
        mt = _robust_mt_scores(data)
        frame = _fast_frame_mt(data)
        ph = _compute_phoneme_scores(data)

        signals = {
            'mt_max':     {p: mt['max'][p] for p in pids},
            'mt_robust':  {p: mt['robust'].get(p, mt['max'][p]) for p in pids},
            'mt_norm':    {p: mt['norm_max'].get(p, mt['max'][p]) for p in pids},
            'mt_quality': {p: mt['quality'].get(p, mt['max'][p]) for p in pids},
            'frame':      {p: frame.get(p, 0.0) for p in pids},
            'phonetic':   {p: ph.get(p, 0.5) for p in pids},
        }

        for name, sig in signals.items():
            s = np.array([sig[p] for p in pids])
            auc = roc_auc(y, s)
            print(f"  {name:12s}: AUC={auc:.4f}")

        # rank 融合
        ranked_signals = {name: rank_norm(sig) for name, sig in signals.items()}
        w_test = {'mt_robust': 0.5, 'mt_quality': 0.2,
                   'frame': 0.15, 'phonetic': 0.05, 'mt_norm': 0.1}
        fused_ranks = {}
        for pid in pids:
            s_val = sum(w * ranked_signals[k].get(pid, 0.5)
                        for k, w in w_test.items())
            fused_ranks[pid] = s_val
        s_fused = np.array([fused_ranks[p] for p in pids])
        auc_fused = roc_auc(y, s_fused)
        print(f"  {'rank_fused':12s}: AUC={auc_fused:.4f}  ***")

        bl = roc_auc(y, np.array([mt['max'][p] for p in pids]))
        print(f"  Baseline max AUC={bl:.4f}, Fused AUC={auc_fused:.4f} "
              f"(+{auc_fused - bl:.4f})")

    print("\nDone!")


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    if '--val' in sys.argv:
        run_val()
    else:
        run_submit()
