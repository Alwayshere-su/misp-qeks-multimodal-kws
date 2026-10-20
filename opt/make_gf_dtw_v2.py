"""gf_split + DTW 权重微调 (基于 submit_gf_dtw.csv 0.90042 的成功方向)
用法: python opt/make_gf_dtw_v2.py
"""
import numpy as np
from pathlib import Path

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def rank_norm(arr):
    order = np.argsort(np.argsort(arr))
    return (order + 1) / len(arr)


def load_sub(name):
    ins, out = {}, {}
    with open(BASE / 'submission' / name, encoding='utf-8') as f:
        for r in f:
            p = r.strip().split(',')
            if len(p) < 3 or p[0] == 'id':
                continue
            (ins if p[1] == 'inset' else out)[int(p[0])] = float(p[2])
    return ins, out


def write(tag_w, out_path):
    gf = load_sub('submit_mt_gf_split.csv')
    with open(BASE / 'submission' / out_path, 'w', encoding='utf-8', newline='') as f:
        f.write('id,type,prediction\n')
        for tag, idx in [('inset', 0), ('outset', 1)]:
            dtw = np.load(BASE / f'opt/_dtw_{tag}_full.npy')
            r_dtw = rank_norm(dtw)
            r_gf = rank_norm(np.array([gf[idx][i] for i in range(50000)]))
            w = tag_w[tag]
            for i in range(50000):
                v = (1 - w) * r_gf[i] + w * r_dtw[i]
                f.write(f'{i},{tag},{v:.6f}\n')
    print(f'saved submission/{out_path} (inset w={tag_w["inset"]}, outset w={tag_w["outset"]})',
          flush=True)


if __name__ == '__main__':
    # v1 基线: inset 0.05, outset 0.3 = 0.90042
    write({'inset': 0.10, 'outset': 0.35}, 'submit_gf_dtw_v2.csv')
    write({'inset': 0.05, 'outset': 0.45}, 'submit_gf_dtw_v3.csv')
    write({'inset': 0.15, 'outset': 0.40}, 'submit_gf_dtw_v4.csv')
