"""最终冲刺提交: 保留 mt_gf_split(0.89995) 的多模板权重, p4 部分替换为独立信号 seq/p1
inset: g0.2 f0.3 p4 0.35 seq 0.075 p1 0.075  (seq/p1 与 mt rho~0.3, 补充)
outset: g0.45 f0.3 p4 0.1 seq 0.075 p1 0.075 (seq/p1 与 mt rho~0, 独立)
"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def load(n):
    ins, out = {}, {}
    for r in csv.DictReader(open(BASE / 'submission' / n, encoding='utf-8')):
        (ins if r['type'] == 'inset' else out)[int(r['id'])] = float(r['prediction'])
    return ins, out


def rn(p):
    ids = sorted(p.keys())
    v = np.array([p[i] for i in ids])
    r = rankdata(v) / len(v)
    return dict(zip(ids, r))


def main():
    g = load('submit_multitemplate.csv')
    f = load('submit_mt_frame_only.csv')
    full = load('submit_p4_full.csv')
    seq = load('submit_seq_foldavg.csv')
    p1 = load('submit_p1_aug_test.csv')
    sig = {'g': g, 'f': f, 'full': full, 'seq': seq, 'p1': p1}
    rn_sig = {k: (rn(v[0]), rn(v[1])) for k, v in sig.items()}

    cfg_i = {'g': 0.20, 'f': 0.30, 'full': 0.35, 'seq': 0.075, 'p1': 0.075}
    cfg_o = {'g': 0.45, 'f': 0.30, 'full': 0.10, 'seq': 0.075, 'p1': 0.075}
    out_path = BASE / 'submission/submit_final_boost.csv'
    with open(out_path, 'w', encoding='utf-8', newline='') as fh:
        fh.write("id,type,prediction\n")
        for i in range(50000):
            sc = sum(cfg_i[k] * rn_sig[k][0].get(i, 0.5) for k in cfg_i)
            fh.write(f"{i},inset,{sc:.6f}\n")
        for i in range(50000):
            sc = sum(cfg_o[k] * rn_sig[k][1].get(i, 0.5) for k in cfg_o)
            fh.write(f"{i},outset,{sc:.6f}\n")
    print(f'saved {out_path.name}', flush=True)
    # 校验
    rows = list(csv.DictReader(open(out_path, encoding='utf-8')))
    ok = len(rows) == 100000 and all(0 <= float(r['prediction']) <= 1 for r in rows)
    print(f'格式校验: rows={len(rows)} ok={ok}', flush=True)


if __name__ == '__main__':
    main()
