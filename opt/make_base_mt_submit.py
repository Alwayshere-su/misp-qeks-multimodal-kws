"""base 多模板 + tiny 多模板 + 帧级 + p4 融合提交"""
import csv
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def base_mt(data, cache):
    """cache: {pid: (e_emb, q_emb)}; 每词 base enroll 模板 -> query max 相似度"""
    from collections import defaultdict
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append(pid)
    scores = {}
    for w, pids in word_items.items():
        T = np.stack([cache[p][0] for p in pids])
        Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
        for p in pids:
            q = cache[p][1]
            qn = q / (np.linalg.norm(q) + 1e-9)
            scores[p] = float((qn @ Tn.T).max())
    return scores


def rn(p):
    ids = sorted(p.keys())
    v = np.array([p[i] for i in ids])
    r = rankdata(v) / len(v)
    return dict(zip(ids, r))


def load(n):
    ins, out = {}, {}
    for r in csv.DictReader(open(BASE / 'submission' / n, encoding='utf-8')):
        (ins if r['type'] == 'inset' else out)[int(r['id'])] = float(r['prediction'])
    return ins, out


def main():
    tin = pickle.load(open(BASE / 'output_whisper/test_inset.pkl', 'rb'))
    tout = pickle.load(open(BASE / 'output_whisper/test_outset.pkl', 'rb'))
    print('base 多模板计算...', flush=True)
    b_i = base_mt(tin, pickle.load(open(BASE / 'output_whisper/test_base_inset.pkl', 'rb')))
    b_o = base_mt(tout, pickle.load(open(BASE / 'output_whisper/test_base_outset.pkl', 'rb')))
    print(f'inset {len(b_i)} outset {len(b_o)}', flush=True)
    rb_i, rb_o = rn(b_i), rn(b_o)

    # tiny 多模板 (g) / 帧级 (f) / p4
    g = load('submit_multitemplate.csv')
    f = load('submit_mt_frame_only.csv')
    full = load('submit_p4_full.csv')
    rg_i, rg_o = rn(g[0]), rn(g[1])
    rf_i, rf_o = rn(f[0]), rn(f[1])
    rfu_i, rfu_o = rn(full[0]), rn(full[1])

    variants = {
        'v1': dict(inset=dict(b=0.1, g=0.1, f=0.3, p=0.5), outset=dict(b=0.2, g=0.3, f=0.3, p=0.2)),
        'v2': dict(inset=dict(b=0.15, g=0.05, f=0.3, p=0.5), outset=dict(b=0.3, g=0.2, f=0.3, p=0.2)),
        'v3': dict(inset=dict(b=0.1, g=0.1, f=0.3, p=0.5), outset=dict(b=0.25, g=0.25, f=0.3, p=0.2)),
    }
    for tag, cfg in variants.items():
        with open(BASE / f'submission/submit_base_mt_{tag}.csv', 'w', encoding='utf-8', newline='') as fh:
            fh.write("id,type,prediction\n")
            for i in range(50000):
                c = cfg['inset']
                sc = c['b'] * rb_i.get(i, 0.5) + c['g'] * rg_i.get(i, 0.5) + \
                     c['f'] * rf_i.get(i, 0.5) + c['p'] * rfu_i.get(i, 0.5)
                fh.write(f"{i},inset,{sc:.6f}\n")
            for i in range(50000):
                c = cfg['outset']
                sc = c['b'] * rb_o.get(i, 0.5) + c['g'] * rg_o.get(i, 0.5) + \
                     c['f'] * rf_o.get(i, 0.5) + c['p'] * rfu_o.get(i, 0.5)
                fh.write(f"{i},outset,{sc:.6f}\n")
        print(f'saved submit_base_mt_{tag}.csv', flush=True)


if __name__ == '__main__':
    main()
