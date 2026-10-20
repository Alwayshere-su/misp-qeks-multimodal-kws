"""val 验证: DTW / fast_frame_mt 与全局多模板、帧级多模板的互补性 + 融合
用法: python opt/_check_dtw_fusion.py
"""
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')
sys.path.insert(0, str(BASE / 'opt'))
from frame_dtw import (constrained_dtw, remove_silence_frames,  # noqa: E402
                       frame_dtw_scores, fast_frame_mt)


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


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


def frame_mt(data):
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
            scores[pid] = float(sim.max(axis=1).mean()) if (sim := qn @ Tn.T).shape else 0.0
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


def rho(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def rank_norm(v):
    return (np.argsort(np.argsort(v)) + 1) / len(v)


def main():
    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))
    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        pids = sorted(data.keys())
        y = np.array([data[p]['label'] for p in pids])
        print(f'\n=== {split} n={len(pids)} pos={y.mean():.3f} ===', flush=True)

        # 全局多模板 / 帧级多模板 / fast_frame_mt (全量)
        g = global_mt(data)
        s_g = np.array([g[p] for p in pids])
        fm = frame_mt(data)
        s_fm = np.array([fm[p] for p in pids])
        ffm = fast_frame_mt(data)
        s_ffm = np.array([ffm[p] for p in pids])

        # DTW 抽样 (2000)
        random.seed(42)
        n_sample = min(2000, len(pids))
        sp = random.sample(pids, n_sample)
        sd = {p: data[p] for p in sp}
        dtw = frame_dtw_scores(sd)
        s_dtw = np.array([dtw[p] for p in sp])
        y_dtw = np.array([data[p]['label'] for p in sp])

        print(f'  global_mt      AUC={roc_auc(y, s_g):.4f}', flush=True)
        print(f'  frame_mt       AUC={roc_auc(y, s_fm):.4f}', flush=True)
        print(f'  fast_frame_mt  AUC={roc_auc(y, s_ffm):.4f}', flush=True)
        print(f'  dtw (sampled)  AUC={roc_auc(y_dtw, s_dtw):.4f}', flush=True)

        # 相关性 (抽样范围内, 与 dtw 对比)
        g_sp = np.array([g[p] for p in sp])
        fm_sp = np.array([fm[p] for p in sp])
        ffm_sp = np.array([ffm[p] for p in sp])
        print(f'  rho: dtw vs global={rho(s_dtw, g_sp):.4f}  '
              f'dtw vs frame_mt={rho(s_dtw, fm_sp):.4f}  '
              f'dtw vs fast_frame_mt={rho(s_dtw, ffm_sp):.4f}', flush=True)
        print(f'       frame_mt vs global={rho(s_fm, s_g):.4f}  '
              f'fast_frame_mt vs frame_mt={rho(s_ffm, s_fm):.4f}  '
              f'fast_frame_mt vs global={rho(s_ffm, s_g):.4f}', flush=True)

        # 融合 (rank): dtw 与 global 融合
        if len(sp) == 2000:
            rg, rd = rank_norm(g_sp), rank_norm(s_dtw)
            for w in [0.1, 0.2, 0.3, 0.5]:
                fused = (1 - w) * rg + w * rd
                print(f'  fuse global({1-w:.1f}) + dtw({w:.1f}): AUC={roc_auc(y_dtw, fused):.4f}',
                      flush=True)


if __name__ == '__main__':
    main()
