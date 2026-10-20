"""分析 foldavg vs full 提交差异: 按词表覆盖(见过/未见)分组"""
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
    fold = load('submit_p4_foldavg_test.csv')
    # test enroll 词表
    seen_words = {}
    for r in csv.DictReader(open(BASE / 'test/eval_seen_enroll_text.csv', encoding='utf-8')):
        seen_words[int(r['id'].split('_')[1]) - 1] = norm(r['enroll_txt'])
    unseen_words = {}
    for r in csv.DictReader(open(BASE / 'test/eval_unseen_enroll_text.csv', encoding='utf-8')):
        unseen_words[int(r['id'].split('_')[1]) - 1] = norm(r['enroll_txt'])
    # val 词表
    val_words = set()
    for split in ['dev_seen', 'dev_unseen']:
        for f in (BASE / f'val/{split}/label').glob('*.txt'):
            parts = f.read_text(encoding='utf-8').strip().split('\t')
            for p in parts:
                if not p.startswith('label'):
                    k, v = p.split(':', 1)
                    val_words.add(norm(v))
    for name, idx, words in [('inset', 0, seen_words), ('outset', 1, unseen_words)]:
        seen_diff, unseen_diff = [], []
        for i in range(50000):
            w = words[i]
            d = fold[idx][i] - full[idx][i]
            (seen_diff if w in val_words else unseen_diff).append(d)
        print(f'{name}: 词在val见过 {len(seen_diff)} 未见 {len(unseen_diff)}', flush=True)
        for tag, arr in [('见过', seen_diff), ('未见', unseen_diff)]:
            a = np.array(arr)
            print(f'  {tag}: fold-full 差异 mean={a.mean():+.4f} |d|>0.2: {(np.abs(a)>0.2).mean():.3f} '
                  f'fold>full: {(a>0.05).mean():.3f} fold<full: {(a<-0.05).mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
