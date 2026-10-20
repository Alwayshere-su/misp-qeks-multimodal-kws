"""ASR 分数 + 声学分数融合评估 (val, 无泄漏协议)
用法: python fuse_eval.py --asr-cache output_whisper/val_asr_small.pkl --acoustic output_p4/oof_scores.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from asr_pipeline import match_score, norm

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def load_val_labels(split):
    labels = {}
    for f in (BASE / f'val/{split}/label').glob('*.txt'):
        parts = f.read_text(encoding='utf-8').strip().split('\t')
        labels[f.stem] = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in parts}
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asr-cache', default='output_whisper/val_asr_small.pkl')
    ap.add_argument('--acoustic', default='output_p4/oof_scores.pkl')
    ap.add_argument('--match', default='edit', choices=['exact', 'edit', 'v2'])
    args = ap.parse_args()

    asr = pickle.load(open(BASE / args.asr_cache, 'rb'))
    aco = pickle.load(open(BASE / args.acoustic, 'rb'))
    print(f'ASR 缓存 {len(asr)} 条, 声学 {sum(len(v) for v in aco.values())} 条', flush=True)

    for split in ['dev_seen', 'dev_unseen']:
        labels = load_val_labels(split)
        pids = sorted(labels.keys())
        y, sa, sc = [], [], []
        for pid in pids:
            if f'{split}/{pid}' not in asr or pid not in aco[split]:
                continue
            txt = asr[f'{split}/{pid}']
            sa.append(match_score(labels[pid]['enrollment'], txt, args.match))
            sc.append(aco[split][pid])
            y.append(1 if labels[pid]['label'] == '1' else 0)
        sa = np.array(sa)
        sc = np.array(sc)
        y = np.array(y)
        print(f'\n=== {split} n={len(y)} pos={y.sum()} ===', flush=True)
        print(f'  ASR alone:  AUC={roc_auc_score(y, sa):.6f}', flush=True)
        print(f'  Acoustic alone: AUC={roc_auc_score(y, sc):.6f}', flush=True)
        # 归一化融合 (rank 或 minmax)
        from scipy.stats import rankdata
        sa_r = rankdata(sa) / len(sa)
        sc_r = rankdata(sc) / len(sc)
        best_w, best_auc = 0.0, 0.0
        for w in np.arange(0.0, 1.01, 0.1):
            fused = w * sa_r + (1 - w) * sc_r
            auc = roc_auc_score(y, fused)
            if auc > best_auc:
                best_auc, best_w = auc, w
            print(f'    w_asr={w:.1f}: AUC={auc:.6f}', flush=True)
        print(f'  BEST w_asr={best_w:.1f} AUC={best_auc:.6f}', flush=True)


if __name__ == '__main__':
    main()
