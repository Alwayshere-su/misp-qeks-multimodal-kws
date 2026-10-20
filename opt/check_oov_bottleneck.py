"""验证未见词瓶颈: p4_full 对见过/未见词的预测分布"""
import csv
import re
from pathlib import Path

import numpy as np

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def load(n):
    ins, out = {}, {}
    for r in csv.DictReader(open(BASE / 'submission' / n, encoding='utf-8')):
        (ins if r['type'] == 'inset' else out)[int(r['id'])] = float(r['prediction'])
    return ins, out


def main():
    full = load('submit_p4_full.csv')
    val_words = set()
    for split in ['dev_seen', 'dev_unseen']:
        for f in (BASE / f'val/{split}/label').glob('*.txt'):
            parts = f.read_text(encoding='utf-8').strip().split('\t')
            for p in parts:
                if not p.startswith('label'):
                    k, v = p.split(':', 1)
                    val_words.add(norm(v))
    for name, idx, csvn in [('inset', 0, 'eval_seen_enroll_text.csv'),
                            ('outset', 1, 'eval_unseen_enroll_text.csv')]:
        words = {}
        for r in csv.DictReader(open(BASE / 'test' / csvn, encoding='utf-8')):
            words[int(r['id'].split('_')[1]) - 1] = norm(r['enroll_txt'])
        seen, unseen = [], []
        for i in range(50000):
            w = words[i]
            (seen if w in val_words else unseen).append(full[idx][i])
        s, u = np.array(seen), np.array(unseen)
        print(f'{name}: 见过词 {len(s)} 未见词 {len(u)}', flush=True)
        print(f'  见过: mean={s.mean():.3f} >0.5:{(s > 0.5).mean():.3f} '
              f'>0.9:{(s > 0.9).mean():.3f}', flush=True)
        print(f'  未见: mean={u.mean():.3f} >0.5:{(u > 0.5).mean():.3f} '
              f'>0.9:{(u > 0.9).mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
