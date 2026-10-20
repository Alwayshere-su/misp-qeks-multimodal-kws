"""检查 train 词表对 test 的覆盖率 (决定 500K 路线价值)"""
import pickle
import re
from pathlib import Path

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def main():
    # train 词表: 用已有的 texts_100000.pkl
    G = BASE / 'output_whisper' / 'train_global'
    train_words = set()
    for n in [100000]:
        tp = G / f'texts_{n}.pkl'
        if tp.exists():
            texts = pickle.load(open(tp, 'rb'))
            for et, qt in texts:
                train_words.add(norm(et))
                train_words.add(norm(qt))
    print(f'train 词表大小(100K label): {len(train_words)}', flush=True)

    import csv
    ts = set(norm(r['enroll_txt']) for r in csv.DictReader(open(BASE / 'test/eval_seen_enroll_text.csv', encoding='utf-8')))
    tu = set(norm(r['enroll_txt']) for r in csv.DictReader(open(BASE / 'test/eval_unseen_enroll_text.csv', encoding='utf-8')))
    print(f'test_seen 词: {len(ts)}  test_unseen 词: {len(tu)}', flush=True)
    ov_s = len(train_words & ts)
    ov_u = len(train_words & tu)
    print(f'train ∩ test_seen: {ov_s}/{len(ts)} ({ov_s / len(ts):.1%})', flush=True)
    print(f'train ∩ test_unseen: {ov_u}/{len(tu)} ({ov_u / len(tu):.1%})', flush=True)

    # val 词表参考
    val_seen = set()
    for f in (BASE / 'val/dev_seen/label').glob('*.txt'):
        parts = f.read_text(encoding='utf-8').strip().split('\t')
        for p in parts:
            if not p.startswith('label'):
                k, v = p.split(':', 1)
                val_seen.add(norm(v))
    print(f'val_seen ∩ test_seen: {len(val_seen & ts)}/{len(ts)}', flush=True)


if __name__ == '__main__':
    main()
