"""验证: test_seen 同一词的多段 query 音频嵌入是否聚类 (无监督提升的基础)"""
import pickle
from pathlib import Path

import numpy as np

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def main():
    tin = pickle.load(open(BASE / 'output_whisper/test_inset.pkl', 'rb'))
    # 统计词分布
    from collections import Counter
    words = Counter(tin[p]['enroll_txt'].lower() for p in tin)
    print(f'test_seen: {len(tin)} pairs, {len(words)} 词', flush=True)
    print('词频分布:', Counter(words.values()).most_common(5), flush=True)

    # 取几个高频词的 query 嵌入, 检查聚类
    w2pairs = {}
    for p, it in tin.items():
        w = it['enroll_txt'].lower()
        w2pairs.setdefault(w, []).append(p)
    for w in ['jeremy', 'cathedral', 'natural', 'hydrogen']:
        pids = w2pairs.get(w, [])[:10]
        if len(pids) < 3:
            continue
        qs = np.stack([tin[p]['query_emb'] for p in pids])
        es = np.stack([tin[p]['enroll_emb'] for p in pids])
        qn = qs / (np.linalg.norm(qs, axis=1, keepdims=True) + 1e-9)
        sim = qn @ qn.T
        # 与 enroll 音频的相似度
        en = es / (np.linalg.norm(es, axis=1, keepdims=True) + 1e-9)
        qe = qn @ en.T
        print(f'词 {w!r} ({len(pids)} pairs):', flush=True)
        print(f'  query-query 相似度 mean(off-diag)={np.mean(sim[~np.eye(len(sim),dtype=bool)]):.3f} '
              f'range=[{sim.min():.2f},{sim.max():.2f}]', flush=True)
        print(f'  query-enroll 相似度 mean={np.mean(qe):.3f} range=[{qe.min():.2f},{qe.max():.2f}]', flush=True)


if __name__ == '__main__':
    main()
