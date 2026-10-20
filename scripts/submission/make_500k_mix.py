"""500k 试探版与 s2u/foldavg 的 rank 平均集成"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def load(path):
    inset, outset = {}, {}
    for r in csv.DictReader(open(path, encoding='utf-8')):
        (inset if r['type'] == 'inset' else outset)[int(r['id'])] = float(r['prediction'])
    return inset, outset


def rank_norm(preds):
    ids = sorted(preds.keys())
    v = np.array([preds[i] for i in ids])
    r = rankdata(v) / len(v)
    return dict(zip(ids, r))


def write_csv(inset, outset, path):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            f.write(f"{i},inset,{inset.get(i, 0.5):.6f}\n")
        for i in range(50000):
            f.write(f"{i},outset,{outset.get(i, 0.5):.6f}\n")
    print(f'-> {path}', flush=True)


def main():
    names = ['submit_500k.csv', 'submit_p4_s2u_test.csv', 'submit_p4_foldavg_test.csv']
    parts = [load(BASE / 'submission' / n) for n in names]
    r_inset = [rank_norm(pt[0]) for pt in parts]
    r_outset = [rank_norm(pt[1]) for pt in parts]
    inset = {i: float(np.mean([r[i] for r in r_inset])) for i in range(50000)}
    outset = {i: float(np.mean([r[i] for r in r_outset])) for i in range(50000)}
    write_csv(inset, outset, str(BASE / 'submission/submit_500k_mix.csv'))
    # 500k + s2u 两两
    ii = {i: 0.5 * (r_inset[0][i] + r_inset[1][i]) for i in range(50000)}
    oo = {i: 0.5 * (r_outset[0][i] + r_outset[1][i]) for i in range(50000)}
    write_csv(ii, oo, str(BASE / 'submission/submit_500k_s2u.csv'))


if __name__ == '__main__':
    main()
