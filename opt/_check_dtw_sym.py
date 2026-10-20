"""val 验证 DTW 变体: 对称双向 / 聚合变体, 找比单向 min 更强的信号
用法: python opt/_check_dtw_sym.py
"""
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')
sys.path.insert(0, str(BASE / 'opt'))
from frame_dtw import constrained_dtw, remove_silence_frames  # noqa: E402


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def dtw_variants(data, window_ratio=0.5):
    """每 pair: 与全部模板做 双向 DTW, 输出多种聚合分数.
    Returns: {pid: {k: score}} k in:
      qe_min (单向 q->e min), eq_min, sym_min, sym_mean, top3_mean, qe_top3_mean
    """
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))

    out = {}
    t0 = time.time()
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
            qe = []
            eq = []
            for ec in tpl_clean:
                d1 = 1.0 - qc @ ec.T
                wsize = max(5, int(window_ratio * max(len(qc), len(ec))))
                v1 = constrained_dtw(d1, wsize)
                d2 = 1.0 - ec @ qc.T
                v2 = constrained_dtw(d2, wsize)
                qe.append(v1)
                eq.append(v2)
            qe = np.array(qe)
            eq = np.array(eq)
            sym = np.minimum(qe, eq)
            out[pid] = {
                'qe_min': -float(qe.min()),
                'eq_min': -float(eq.min()),
                'sym_min': -float(sym.min()),
                'sym_mean': -float(np.sort(sym)[:3].mean()) if len(sym) >= 3 else -float(sym.mean()),
                'qe_top3': -float(np.sort(qe)[:3].mean()) if len(qe) >= 3 else -float(qe.mean()),
            }
    return out


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


def global_mt(data):
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))
    scores = {}
    for w, items in word_items.items():
        T = np.stack([it['enroll_emb'] for _, it in items])
        Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
        for pid, it in items:
            q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
            scores[pid] = float((q @ Tn.T).max())
    return scores


def rho(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def main():
    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))
    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        pids = sorted(data.keys())
        random.seed(42)
        n_sample = min(3000, len(pids))
        sp = random.sample(pids, n_sample)
        sd = {p: data[p] for p in sp}
        y = np.array([data[p]['label'] for p in sp])
        t0 = time.time()
        dv = dtw_variants(sd)
        print(f'\n=== {split} (抽 {n_sample}, {time.time()-t0:.0f}s) ===', flush=True)
        g = global_mt(data)
        g_sp = np.array([g[p] for p in sp])
        for k in ['qe_min', 'eq_min', 'sym_min', 'sym_mean', 'qe_top3']:
            s = np.array([dv[p][k] for p in sp])
            auc = roc_auc(y, s)
            # 融合 global (w=0.3 同 v1 成功配置)
            r_s = (np.argsort(np.argsort(s)) + 1) / len(s)
            r_g = (np.argsort(np.argsort(g_sp)) + 1) / len(g_sp)
            fused = 0.7 * r_g + 0.3 * r_s
            print(f'  {k:10s}: AUC={auc:.4f}  rho(dtw,global)={rho(s, g_sp):.4f}  '
                  f'fuse(0.7g+0.3d)={roc_auc(y, fused):.4f}', flush=True)


if __name__ == '__main__':
    main()
