"""全模板 DTW test 分数生成 + 融合 gf_split (outset 加 DTW, val 证据 unseen 融合有效)
用法:
  python opt/make_dtw_submit.py --val-only   # val 全模板 DTW 验证 (对照 0.9770)
  python opt/make_dtw_submit.py              # test 全量 + 融合提交 (后台, 30-50min)
"""
import argparse
import os
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get('MISP_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(BASE / 'opt'))
from frame_dtw import constrained_dtw, remove_silence_frames  # noqa: E402


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def dtw_full(data, log_every=5000):
    """全模板 DTW: 每词模板帧只 clean/归一化一次, 逐 pair 约束 DTW 取最小距离.
    Returns: {pid: -best_dtw} (越大越好)
    """
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))

    scores = {}
    t0 = time.time()
    n_done = 0
    n_dtw = 0
    for w, items in word_items.items():
        tpl_clean = []
        for _, it_t in items:
            ef = it_t['enroll_frames'].astype(np.float32)
            ec, _ = remove_silence_frames(ef)
            ec = ec / (np.linalg.norm(ec, axis=1, keepdims=True) + 1e-9)
            tpl_clean.append(ec)
        for pid, it in items:
            qf = it['query_frames'].astype(np.float32)
            qc, _ = remove_silence_frames(qf)
            qc = qc / (np.linalg.norm(qc, axis=1, keepdims=True) + 1e-9)
            best = np.inf
            for ec in tpl_clean:
                dist = 1.0 - qc @ ec.T
                w_size = max(5, int(0.5 * max(len(qc), len(ec))))
                v = constrained_dtw(dist, w_size)
                n_dtw += 1
                if v < best:
                    best = v
            scores[pid] = -float(best)
            n_done += 1
        if n_done % log_every == 0:
            print(f'  ...{n_done} pairs ({time.time()-t0:.0f}s, dtw_calls={n_dtw})', flush=True)
    return scores


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


def rank_norm(d):
    ids = sorted(d.keys())
    v = np.array([d[i] for i in ids], dtype=np.float64)
    order = np.argsort(np.argsort(v))
    return dict(zip(ids, (order + 1) / len(v)))


def load_sub(name):
    ins, out = {}, {}
    with open(BASE / 'submission' / name, encoding='utf-8') as f:
        for r in f:
            p = r.strip().split(',')
            if len(p) < 3 or p[0] == 'id':
                continue
            (ins if p[1] == 'inset' else out)[int(p[0])] = float(p[2])
    return ins, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-only', action='store_true')
    args = ap.parse_args()

    if args.val_only:
        vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))
        for split in ['dev_seen', 'dev_unseen']:
            data = vd[split]
            pids = sorted(data.keys())
            y = np.array([data[p]['label'] for p in pids])
            random.seed(42)
            n_sample = min(3000, len(pids))
            sp = random.sample(pids, n_sample)
            sd = {p: data[p] for p in sp}
            t0 = time.time()
            dtw = dtw_full(sd, log_every=100000)
            s = np.array([dtw[p] for p in sp])
            yy = np.array([data[p]['label'] for p in sp])
            print(f'=== {split} 全模板 DTW (抽 {n_sample}): AUC={roc_auc(yy, s):.4f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
        return

    # ---- test 全量 ----
    for tag, feats in [('inset', 'test_inset.pkl'), ('outset', 'test_outset.pkl')]:
        print(f'加载 {feats} ...', flush=True)
        t0 = time.time()
        d = pickle.load(open(BASE / 'output_whisper' / feats, 'rb'))
        print(f'  {len(d)} pairs, {time.time()-t0:.0f}s', flush=True)
        t0 = time.time()
        dtw = dtw_full(d)
        arr = np.array([dtw[f'pair_{i+1:06d}'] for i in range(50000)])
        np.save(BASE / f'opt/_dtw_{tag}_full.npy', arr)
        print(f'  {tag}: DTW 完成 ({time.time()-t0:.0f}s), saved _dtw_{tag}_full.npy', flush=True)

    # ---- 融合 gf_split ----
    gf = load_sub('submit_mt_gf_split.csv')
    # val 证据: unseen 融合 global+dtw(0.3) 有效; seen 无效/微降 -> outset 0.3, inset 0.05
    weights = {'inset': 0.05, 'outset': 0.3}
    with open(BASE / 'submission/submit_gf_dtw.csv', 'w', encoding='utf-8', newline='') as f:
        f.write('id,type,prediction\n')
        for tag, idx in [('inset', 0), ('outset', 1)]:
            arr = np.load(BASE / f'opt/_dtw_{tag}_full.npy')
            r_dtw = rank_norm({i: float(v) for i, v in enumerate(arr)})
            r_gf = rank_norm(gf[idx])
            w = weights[tag]
            fused = {i: (1 - w) * r_gf.get(i, 0.5) + w * r_dtw.get(i, 0.5) for i in range(50000)}
            for i in range(50000):
                f.write(f'{i},{tag},{fused[i]:.6f}\n')
    print('saved submission/submit_gf_dtw.csv', flush=True)


if __name__ == '__main__':
    main()
