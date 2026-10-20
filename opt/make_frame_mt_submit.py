"""帧级多模板 test 提交 + 与全局多模板融合 (无监督组合)"""
import csv
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def global_mt(data):
    from collections import defaultdict
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
    from collections import defaultdict
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
            sim = qn @ Tn.T
            scores[pid] = float(sim.max(axis=1).mean())
    return scores


def rn(p):
    ids = sorted(p.keys())
    v = np.array([p[i] for i in ids])
    r = rankdata(v) / len(v)
    return dict(zip(ids, r))


def main():
    print('加载 test 特征...', flush=True)
    tin = pickle.load(open(BASE / 'output_whisper/test_inset.pkl', 'rb'))
    tout = pickle.load(open(BASE / 'output_whisper/test_outset.pkl', 'rb'))
    print('全局多模板...', flush=True)
    gi = global_mt(tin)
    go = global_mt(tout)
    print('帧级多模板...', flush=True)
    fi = frame_mt(tin)
    fo = frame_mt(tout)
    print('融合生成...', flush=True)
    rgi, rgo = rn(gi), rn(go)
    rfi, rfo = rn(fi), rn(fo)
    for wg in [0.3, 0.5, 0.7]:
        with open(BASE / f'submission/submit_mt_frame_w{wg:.1f}.csv', 'w', encoding='utf-8', newline='') as f:
            f.write("id,type,prediction\n")
            for i in range(50000):
                pid = f"pair_{i+1:06d}"
                sc = wg * rgi.get(pid, 0.5) + (1 - wg) * rfi.get(pid, 0.5)
                f.write(f"{i},inset,{sc:.6f}\n")
            for i in range(50000):
                pid = f"pair_{i+1:06d}"
                sc = wg * rgo.get(pid, 0.5) + (1 - wg) * rfo.get(pid, 0.5)
                f.write(f"{i},outset,{sc:.6f}\n")
        print(f'saved submit_mt_frame_w{wg:.1f}.csv', flush=True)
    # 帧级纯版
    with open(BASE / 'submission/submit_mt_frame_only.csv', 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},inset,{rfi.get(pid, 0.5):.6f}\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},outset,{rfo.get(pid, 0.5):.6f}\n")
    print('saved submit_mt_frame_only.csv', flush=True)


if __name__ == '__main__':
    main()
