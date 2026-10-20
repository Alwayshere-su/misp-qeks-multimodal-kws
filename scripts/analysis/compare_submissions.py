"""对比各提交预测的相关性 (判断差异度)"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def load(path):
    inset, outset = {}, {}
    for r in csv.DictReader(open(path, encoding='utf-8')):
        (inset if r['type'] == 'inset' else outset)[int(r['id'])] = float(r['prediction'])
    return inset, outset


def main():
    names = ['submit_p1p4_ensemble5.csv', 'submit_p4_s2u_test.csv',
             'submit_p4_foldavg_test.csv', 'submit_s2u_foldavg.csv',
             'submit_combo4_test.csv', 'submit_p4_full.csv']
    data = {n: load(BASE / 'submission' / n) for n in names}
    print('Spearman 相关 (inset / outset):')
    base = data['submit_p1p4_ensemble5.csv']
    for n in names[1:]:
        for split, idx in [('inset', 0), ('outset', 1)]:
            a = np.array([base[idx][i] for i in range(50000)])
            b = np.array([data[n][idx][i] for i in range(50000)])
            rho = spearmanr(a, b).statistic
            print(f'  p1p4_ensemble5 vs {n}: {split} rho={rho:.4f}', flush=True)


if __name__ == '__main__':
    main()
