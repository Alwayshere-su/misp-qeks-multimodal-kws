"""验证: train 前 50K pairs (试探版) 的词表对 test_seen 的覆盖"""
import csv
import re
from pathlib import Path

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def main():
    # 读 train 前 50000 的 label 词表 (直接读 txt, 抽样可能慢, 用已有的 texts_50000? 无; 读 20000 估算)
    words_20k = set()
    words_50k = set()
    for i in range(1, 20001):
        parts = open(BASE / 'train/label' / f'pair_{i:06d}.txt', encoding='utf-8').read().strip().split('\t')
        for p in parts:
            if not p.startswith('label'):
                k, v = p.split(':', 1)
                words_20k.add(norm(v))
    ts = set(norm(r['enroll_txt']) for r in csv.DictReader(open(BASE / 'test/eval_seen_enroll_text.csv', encoding='utf-8')))
    print(f'train 前20K 词表: {len(words_20k)}, 覆盖 test_seen: {len(words_20k & ts)}/{len(ts)} '
          f'({len(words_20k & ts) / len(ts):.1%})', flush=True)


if __name__ == '__main__':
    main()
