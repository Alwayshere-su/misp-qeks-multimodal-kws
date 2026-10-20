"""验证 train (chunk0) 嵌入与 test 嵌入的分布一致性"""
import pickle
from pathlib import Path

import numpy as np

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def main():
    G = BASE / 'output_whisper' / 'train_global'
    te = np.load(G / 'enroll_00.npy')
    tq = np.load(G / 'query_00.npy')
    print(f'train chunk0: enroll {te.shape} query {tq.shape}', flush=True)
    tin = pickle.load(open(BASE / 'output_whisper/test_inset.pkl', 'rb'))
    tout = pickle.load(open(BASE / 'output_whisper/test_outset.pkl', 'rb'))
    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))

    def stats(arr, name):
        n = np.linalg.norm(arr, axis=1)
        print(f'  {name}: norm_mean={n.mean():.2f} mean_vec={np.linalg.norm(arr.mean(0)):.2f}', flush=True)
        return arr.mean(0)

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    groups = {
        'train_enroll': te, 'train_query': tq,
        'test_seen_enroll': np.stack([tin[p]['enroll_emb'] for p in sorted(tin)[:20000]]),
        'test_seen_query': np.stack([tin[p]['query_emb'] for p in sorted(tin)[:20000]]),
        'test_unseen_enroll': np.stack([tout[p]['enroll_emb'] for p in sorted(tout)[:20000]]),
        'test_unseen_query': np.stack([tout[p]['query_emb'] for p in sorted(tout)[:20000]]),
        'val_seen_query': np.stack([vd['dev_seen'][p]['query_emb'] for p in sorted(vd['dev_seen'])[:5000]]),
        'val_unseen_query': np.stack([vd['dev_unseen'][p]['query_emb'] for p in sorted(vd['dev_unseen'])[:5000]]),
    }
    means = {}
    for name, arr in groups.items():
        means[name] = stats(arr, name)
    print('质心余弦:', flush=True)
    pairs = [('train_query', 'test_seen_query'), ('train_query', 'test_unseen_query'),
             ('train_query', 'val_seen_query'), ('train_query', 'val_unseen_query'),
             ('train_enroll', 'test_seen_enroll'), ('train_enroll', 'test_unseen_enroll'),
             ('test_seen_query', 'test_unseen_query')]
    for a, b in pairs:
        print(f'  {a} vs {b}: {cos(means[a], means[b]):.4f}', flush=True)


if __name__ == '__main__':
    main()
