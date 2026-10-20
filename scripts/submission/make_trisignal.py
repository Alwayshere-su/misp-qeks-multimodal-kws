"""三信号融合提交: 多模板 + p4_full + seq (互补信号最大化)"""
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


def rank_norm(preds):
    ids = sorted(preds.keys())
    v = np.array([preds[i] for i in ids])
    r = rankdata(v) / len(v)
    return dict(zip(ids, r))


def main():
    mt = load('submit_multitemplate.csv')
    full = load('submit_p4_full.csv')
    seq = load('submit_seq_foldavg.csv')
    # 相关性
    for name, d in [('full', full), ('seq', seq)]:
        for split, idx in [('inset', 0), ('outset', 1)]:
            x = np.array([mt[idx][i] for i in range(50000)])
            y = np.array([d[idx][i] for i in range(50000)])
            print(f'mt vs {name} {split}: rho={spearmanr(x, y).statistic:.4f}', flush=True)
    # 融合
    r_mt_i, r_mt_o = rank_norm(mt[0]), rank_norm(mt[1])
    r_f_i, r_f_o = rank_norm(full[0]), rank_norm(full[1])
    r_s_i, r_s_o = rank_norm(seq[0]), rank_norm(seq[1])
    # 变体 1: mt*0.5 + full*0.3 + seq*0.2
    for tag, wi, wf, ws in [('mt5_f3_s2', 0.5, 0.3, 0.2),
                            ('mt4_f4_s2', 0.4, 0.4, 0.2),
                            ('mt6_f4', 0.6, 0.4, 0.0)]:
        with open(BASE / f'submission/submit_{tag}.csv', 'w', encoding='utf-8', newline='') as f:
            f.write("id,type,prediction\n")
            for i in range(50000):
                sc = wi * r_mt_i[i] + wf * r_f_i[i] + ws * r_s_i[i]
                f.write(f"{i},inset,{sc:.6f}\n")
            for i in range(50000):
                sc = wi * r_mt_o[i] + wf * r_f_o[i] + ws * r_s_o[i]
                f.write(f"{i},outset,{sc:.6f}\n")
        print(f'saved submit_{tag}.csv', flush=True)


if __name__ == '__main__':
    main()
