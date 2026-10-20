"""分析 val 与 test 关键词分布重叠"""
import csv
import re
from pathlib import Path

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def main():
    val_seen, val_unseen = set(), set()
    for split, target in [('dev_seen', val_seen), ('dev_unseen', val_unseen)]:
        for f in (BASE / f'val/{split}/label').glob('*.txt'):
            parts = f.read_text(encoding='utf-8').strip().split('\t')
            d = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in parts}
            target.add(norm(d['query']))
            target.add(norm(d['enrollment']))
    test_seen = [norm(r['enroll_txt']) for r in
                 csv.DictReader(open(BASE / 'test/eval_seen_enroll_text.csv', encoding='utf-8'))]
    test_unseen = [norm(r['enroll_txt']) for r in
                   csv.DictReader(open(BASE / 'test/eval_unseen_enroll_text.csv', encoding='utf-8'))]
    print('val_seen 词数:', len(val_seen), '| val_unseen 词数:', len(val_unseen))
    print('test_seen 唯一词数:', len(set(test_seen)), '| test_unseen 唯一词数:', len(set(test_unseen)))
    print('test_seen ∩ val_seen:', len(set(test_seen) & val_seen),
          '| test_seen ∩ val_unseen:', len(set(test_seen) & val_unseen))
    print('test_unseen ∩ val_unseen:', len(set(test_unseen) & val_unseen),
          '| test_unseen ∩ val_seen:', len(set(test_unseen) & val_seen))


if __name__ == '__main__':
    main()
