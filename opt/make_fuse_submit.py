"""基于最优 gf_split + 新独立信号 (500K 模型, seq 帧级) 的 split 感知融合
逻辑:
  - inset: 融合有效 -> 给新信号更多权重
  - outset: max/robust 主导 -> 保持 gf_split 高权重, 新信号少量补充
变体只调新信号权重这一维度, 便于提交反馈定位
用法: python opt/make_fuse_submit.py
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
    """split 内独立 rank -> [0,1]"""
    ids = sorted(d.keys())
    v = np.array([d[i] for i in ids], dtype=np.float64)
    order = np.argsort(np.argsort(v))
    r = (order + 1) / len(v)
    return dict(zip(ids, r))


def fuse(ins_sigs, out_sigs, out_path):
    """ins_sigs / out_sigs: {name: {pid: raw}} + 权重 dict"""
    weighted = {}
    for split, sigs in [('inset', ins_sigs), ('outset', out_sigs)]:
        ranked = {name: rank_norm(d) for name, d in sigs['signals'].items()}
        w = sigs['weights']
        fused = {}
        for i in range(50000):
            s = sum(w[n] * ranked[n].get(i, 0.5) for n in w if n in ranked)
            ws = sum(w[n] for n in w if n in ranked)
            fused[i] = s / ws if ws > 0 else 0.5
        weighted[split] = fused
    with open(BASE / 'submission' / out_path, 'w', encoding='utf-8', newline='') as f:
        f.write('id,type,prediction\n')
        for i in range(50000):
            f.write(f'{i},inset,{weighted["inset"][i]:.6f}\n')
        for i in range(50000):
            f.write(f'{i},outset,{weighted["outset"][i]:.6f}\n')
    print(f'saved submission/{out_path}', flush=True)
    # 分布检查
    for split in ['inset', 'outset']:
        v = np.array([weighted[split][i] for i in range(50000)])
        print(f'  {split}: mean={v.mean():.3f} std={v.std():.3f} '
              f'>0.9={(v > 0.9).mean():.3f} >0.8={(v > 0.8).mean():.3f}', flush=True)


def main():
    gf = load('submit_mt_gf_split.csv')
    k = load('submit_500k.csv')
    q = load('submit_seq_foldavg.csv')

    # 变体 1: 温和 — 新信号各 0.15/0.10
    fuse(
        {'signals': {'gf': gf[0], 'k': k[0], 'q': q[0]}, 'weights': {'gf': 0.75, 'k': 0.15, 'q': 0.10}},
        {'signals': {'gf': gf[1], 'k': k[1], 'q': q[1]}, 'weights': {'gf': 0.75, 'k': 0.15, 'q': 0.10}},
        'submit_fuse_v1.csv')

    # 变体 2: outset 更保守 (gf 0.8), inset 同 v1
    fuse(
        {'signals': {'gf': gf[0], 'k': k[0], 'q': q[0]}, 'weights': {'gf': 0.75, 'k': 0.15, 'q': 0.10}},
        {'signals': {'gf': gf[1], 'k': k[1], 'q': q[1]}, 'weights': {'gf': 0.80, 'k': 0.15, 'q': 0.05}},
        'submit_fuse_v2.csv')

    # 变体 3: 激进试 500K — inset k 0.30, outset k 0.20
    fuse(
        {'signals': {'gf': gf[0], 'k': k[0], 'q': q[0]}, 'weights': {'gf': 0.60, 'k': 0.30, 'q': 0.10}},
        {'signals': {'gf': gf[1], 'k': k[1], 'q': q[1]}, 'weights': {'gf': 0.70, 'k': 0.20, 'q': 0.10}},
        'submit_fuse_v3.csv')


if __name__ == '__main__':
    main()
