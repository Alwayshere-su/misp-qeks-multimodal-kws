"""seq(帧级) + full 集成 + 多种组合生成"""
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
    full = load('submit_p4_full.csv')
    seq = load('submit_seq_foldavg.csv')
    p1p4 = load('submit_p1p4_ensemble5.csv')
    r_full_i = rank_norm(full[0]); r_full_o = rank_norm(full[1])
    r_seq_i = rank_norm(seq[0]); r_seq_o = rank_norm(seq[1])
    r_p1p4_i = rank_norm(p1p4[0]); r_p1p4_o = rank_norm(p1p4[1])

    # seq + full 各半
    ii = {i: 0.5 * (r_seq_i[i] + r_full_i[i]) for i in range(50000)}
    oo = {i: 0.5 * (r_seq_o[i] + r_full_o[i]) for i in range(50000)}
    write_csv(ii, oo, str(BASE / 'submission/submit_seq_full_half.csv'))

    # seq + full 加权 (seq 0.3, full 0.7)
    ii = {i: 0.3 * r_seq_i[i] + 0.7 * r_full_i[i] for i in range(50000)}
    oo = {i: 0.3 * r_seq_o[i] + 0.7 * r_full_o[i] for i in range(50000)}
    write_csv(ii, oo, str(BASE / 'submission/submit_seq_full_w03.csv'))

    # seq + p1p4
    ii = {i: 0.5 * (r_seq_i[i] + r_p1p4_i[i]) for i in range(50000)}
    oo = {i: 0.5 * (r_seq_o[i] + r_p1p4_o[i]) for i in range(50000)}
    write_csv(ii, oo, str(BASE / 'submission/submit_seq_p1p4_half.csv'))

    # 三者平均
    ii = {i: (r_seq_i[i] + r_full_i[i] + r_p1p4_i[i]) / 3 for i in range(50000)}
    oo = {i: (r_seq_o[i] + r_full_o[i] + r_p1p4_o[i]) / 3 for i in range(50000)}
    write_csv(ii, oo, str(BASE / 'submission/submit_seq_full_p1p4_avg.csv'))


if __name__ == '__main__':
    main()
