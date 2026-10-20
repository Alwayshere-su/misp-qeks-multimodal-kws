"""验证 val pair 是否按难度排序"""
from pathlib import Path

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')

segments = [(1, 10), (500, 510), (1000, 1010), (2000, 2010), (3000, 3010),
            (4000, 4010), (4990, 5000)]
for split in ['dev_seen', 'dev_unseen']:
    print(f'=== {split} ===')
    for s, e in segments:
        words = []
        for i in range(s, e + 1):
            f = BASE / f'val/{split}/label/pair_{i:06d}.txt'
            if not f.exists():
                continue
            parts = f.read_text(encoding='utf-8').strip().split('\t')
            d = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in parts}
            words.append(d['query'])
        print(f'  pair_{s:06d}..{e:06d}: {words[:10]}')
