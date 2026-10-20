"""组合集成提交: s2u + foldavg + full + p1 的 rank 平均"""
import csv
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

BASE = Path(r"D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")


def write_csv(inset, outset, path):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            pid = f"pair_{i + 1:06d}"
            f.write(f"{i},inset,{inset.get(pid, 0.5):.6f}\n")
        for i in range(50000):
            pid = f"pair_{i + 1:06d}"
            f.write(f"{i},outset,{outset.get(pid, 0.5):.6f}\n")
    print(f"  -> {path}")


def load_csv(path):
    out = {}
    for r in csv.DictReader(open(path, encoding='utf-8')):
        typ = r['type']
        pid = f"pair_{int(r['id']) + 1:06d}"
        out.setdefault(typ, {})[pid] = float(r['prediction'])
    return out.get('inset', {}), out.get('outset', {})


def rank_norm(preds):
    pids = sorted(preds.keys())
    v = np.array([preds[p] for p in pids])
    return dict(zip(pids, rankdata(v) / len(v)))


def main():
    names = ['submit_p4_s2u_test.csv', 'submit_p4_foldavg_test.csv',
             'submit_p4_full.csv', 'submit_p1_aug_test.csv']
    parts = []
    for name in names:
        t0 = time.time()
        ii, oo = load_csv(BASE / 'submission' / name)
        print(f'{name}: inset {len(ii)} outset {len(oo)} load={time.time() - t0:.1f}s', flush=True)
        t0 = time.time()
        parts.append((rank_norm(ii), rank_norm(oo)))
        print(f'  rank done {time.time() - t0:.1f}s', flush=True)
    inset = {p: float(np.mean([pt[0][p] for pt in parts])) for p in parts[0][0]}
    outset = {p: float(np.mean([pt[1][p] for pt in parts])) for p in parts[0][1]}
    write_csv(inset, outset, str(BASE / 'submission/submit_combo4_test.csv'))
    # s2u + foldavg 两两
    ii_a, oo_a = load_csv(BASE / 'submission/submit_p4_s2u_test.csv')
    ii_b, oo_b = load_csv(BASE / 'submission/submit_p4_foldavg_test.csv')
    ra = rank_norm(ii_a); rb = rank_norm(ii_b)
    inset = {p: 0.5 * (ra[p] + rb[p]) for p in ii_a}
    ra2 = rank_norm(oo_a); rb2 = rank_norm(oo_b)
    outset = {p: 0.5 * (ra2[p] + rb2[p]) for p in oo_a}
    write_csv(inset, outset, str(BASE / 'submission/submit_s2u_foldavg.csv'))


if __name__ == '__main__':
    main()
