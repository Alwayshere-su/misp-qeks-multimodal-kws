"""归因实验: 分别隔离 500k 与 seq, 定位 v1 (0.8934) 掉分主因
  - iso_500k: gf 0.85 + 500k 0.15   (无 seq)
  - iso_seq:  gf 0.85 + seq 0.15    (无 500k)
两个 split 对称权重, 便于与 gf_split (0.89995) 直接对比
用法: python opt/make_iso_submit.py
"""
import csv
from pathlib import Path

import numpy as np

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def load(name):
    ins, out = {}, {}
    with open(BASE / 'submission' / name, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            (ins if r['type'] == 'inset' else out)[int(r['id'])] = float(r['prediction'])
    return ins, out


def rank_norm(d):
    ids = sorted(d.keys())
    v = np.array([d[i] for i in ids], dtype=np.float64)
    order = np.argsort(np.argsort(v))
    r = (order + 1) / len(v)
    return dict(zip(ids, r))


def fuse(ins_sigs, out_sigs, weights, out_path):
    """ins_sigs/out_sigs: {name: {pid: raw}}; 分别 rank 融合 inset 和 outset"""
    ranked_ins = {name: rank_norm(d) for name, d in ins_sigs.items()}
    ranked_out = {name: rank_norm(d) for name, d in out_sigs.items()}
    with open(BASE / 'submission' / out_path, 'w', encoding='utf-8', newline='') as f:
        f.write('id,type,prediction\n')
        for tag, ranked in [('inset', ranked_ins), ('outset', ranked_out)]:
            for i in range(50000):
                s = sum(w * ranked[n].get(i, 0.5) for n, w in weights.items() if n in ranked)
                ws = sum(w for n, w in weights.items() if n in ranked)
                f.write(f'{i},{tag},{s / ws if ws > 0 else 0.5:.6f}\n')
    print(f'saved submission/{out_path}', flush=True)


def main():
    gf = load('submit_mt_gf_split.csv')
    k = load('submit_500k.csv')
    q = load('submit_seq_foldavg.csv')

    # A: gf + 500k (无 seq), inset/outset 各自独立融合
    fuse({'gf': gf[0], 'k': k[0]}, {'gf': gf[1], 'k': k[1]},
         {'gf': 0.85, 'k': 0.15}, 'submit_iso_500k.csv')
    # B: gf + seq (无 500k)
    fuse({'gf': gf[0], 'q': q[0]}, {'gf': gf[1], 'q': q[1]},
         {'gf': 0.85, 'q': 0.15}, 'submit_iso_seq.csv')


if __name__ == '__main__':
    main()
