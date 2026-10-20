"""分析关键提交信号的分布 + 与最优 (submit_mt_gf_split) 的相关性
用法: python opt/_analyze_signals.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def load(name):
    ins, out = {}, {}
    with open(BASE / 'submission' / name, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            (ins if r['type'] == 'inset' else out)[int(r['id'])] = float(r['prediction'])
    return ins, out


def to_arr(d):
    return np.array([d[i] for i in range(50000)], dtype=np.float64)


def fast_rho(x, y):
    # Spearman = Pearson on ranks
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry) / (np.sqrt(rx @ rx) * np.sqrt(ry @ ry)))


def main():
    files = [
        'submit_mt_gf_split.csv',                    # 当前最优
        'submit_500k.csv', 'submit_500k_v2.csv',     # 500K 模型
        'submit_500k_mix.csv', 'submit_500k_s2u.csv',
        'submit_mix500k_inset_p4full_outset.csv',
        'submit_base_mt_v1.csv', 'submit_base_mt_v2.csv', 'submit_base_mt_v3.csv',
        'submit_multitemplate.csv', 'submit_mt_frame_only.csv',
        'submit_mt_robust_only.csv', 'submit_robust_mt.csv',
        'submit_robust_mt_p4_v4.csv',
        'submit_p4_full.csv', 'submit_p4_foldavg_test.csv',
        'submit_seq_foldavg.csv', 'submit_seq_full_p1p4_avg.csv',
    ]
    data = {}
    for n in files:
        try:
            data[n] = (to_arr(load(n)[0]), to_arr(load(n)[1]))
        except Exception as e:
            print(f'!! {n} 加载失败: {e}', flush=True)
    print(f'成功加载 {len(data)} 个文件\n', flush=True)

    names = sorted(data.keys())
    print('=' * 110, flush=True)
    print('分布统计:  inset / outset  (mean, >0.9 比例, std)', flush=True)
    print('=' * 110, flush=True)
    for n in names:
        a, b = data[n]
        parts = []
        for v, tag in [(a, 'inset'), (b, 'outset')]:
            parts.append(f"{tag} m={v.mean():.3f} >0.9={(v > 0.9).mean():.3f} "
                         f">0.8={(v > 0.8).mean():.3f} std={v.std():.3f}")
        print(f"{n}\n    {' | '.join(parts)}", flush=True)

    ref = data['submit_mt_gf_split.csv']
    print('\n' + '=' * 110, flush=True)
    print('与 submit_mt_gf_split 的 Spearman 相关 (split 独立)', flush=True)
    print('=' * 110, flush=True)
    for n in names:
        if n == 'submit_mt_gf_split.csv':
            continue
        a, b = data[n]
        print(f"{n:48s} inset_rho={fast_rho(ref[0], a):.4f}  "
              f"outset_rho={fast_rho(ref[1], b):.4f}", flush=True)

    print('\n' + '=' * 110, flush=True)
    print('关键互补对相关性', flush=True)
    print('=' * 110, flush=True)
    pairs = [
        ('submit_500k.csv', 'submit_base_mt_v1.csv'),
        ('submit_500k.csv', 'submit_multitemplate.csv'),
        ('submit_500k.csv', 'submit_mt_frame_only.csv'),
        ('submit_base_mt_v1.csv', 'submit_multitemplate.csv'),
        ('submit_base_mt_v1.csv', 'submit_mt_frame_only.csv'),
        ('submit_mt_frame_only.csv', 'submit_multitemplate.csv'),
        ('submit_500k_v2.csv', 'submit_500k.csv'),
        ('submit_500k.csv', 'submit_p4_full.csv'),
        ('submit_500k.csv', 'submit_seq_foldavg.csv'),
    ]
    for x, y in pairs:
        if x not in data or y not in data:
            print(f'{x} vs {y}: 缺失', flush=True)
            continue
        print(f"{x} vs {y}: inset={fast_rho(data[x][0], data[y][0]):.4f} "
              f"outset={fast_rho(data[x][1], data[y][1]):.4f}", flush=True)

    # 保存为 npz 供融合脚本复用
    np.savez(BASE / 'opt/_signals.npz',
             **{n.replace('.csv', '').replace('submit_', 's_'): np.concatenate([data[n][0], data[n][1]])
                for n in data})
    print('\nsaved opt/_signals.npz', flush=True)


if __name__ == '__main__':
    main()
