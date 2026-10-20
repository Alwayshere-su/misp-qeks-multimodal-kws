"""五信号融合: 全局mt + 帧级mt + p4 + seq + p1 (rank 平均, 修复 O(n^2))"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

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
    names = {
        'g': 'submit_multitemplate.csv',
        'f': 'submit_mt_frame_only.csv',
        'full': 'submit_p4_full.csv',
        'seq': 'submit_seq_foldavg.csv',
        'p1': 'submit_p1_aug_test.csv',
    }
    sig = {k: load(v) for k, v in names.items()}
    rn_sig = {k: (rn(v[0]), rn(v[1])) for k, v in sig.items()}

    # 相关性
    ids = sorted(sig['g'][0].keys())
    mt_arr = np.array([rn_sig['g'][0][i] for i in ids])
    for k in ['seq', 'p1']:
        for typ, idx in [('inset', 0), ('outset', 1)]:
            arr = np.array([rn_sig[k][idx][i] for i in ids])
            print(f'mt vs {k} {typ}: rho={spearmanr(mt_arr, arr).statistic:.4f}', flush=True)

    # 五信号融合
    cfg_i = {'g': 0.15, 'f': 0.20, 'full': 0.35, 'seq': 0.15, 'p1': 0.15}
    cfg_o = {'g': 0.30, 'f': 0.20, 'full': 0.15, 'seq': 0.20, 'p1': 0.15}
    with open(BASE / 'submission/submit_five_sig.csv', 'w', encoding='utf-8', newline='') as fh:
        fh.write("id,type,prediction\n")
        for i in range(50000):
            sc = sum(cfg_i[k] * rn_sig[k][0].get(i, 0.5) for k in cfg_i)
            fh.write(f"{i},inset,{sc:.6f}\n")
        for i in range(50000):
            sc = sum(cfg_o[k] * rn_sig[k][1].get(i, 0.5) for k in cfg_o)
            fh.write(f"{i},outset,{sc:.6f}\n")
    print('saved submit_five_sig.csv', flush=True)


if __name__ == '__main__':
    main()
