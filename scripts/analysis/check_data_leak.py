"""快速数据泄漏检测: 先按文件大小粗筛, 仅对相同大小的跨集文件对做 md5"""
import hashlib
import pickle
from pathlib import Path

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def md5_of(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 1. 收集文件大小
    groups = {
        'val_seen_q': (BASE / 'val/dev_seen/wav').glob('*_query.wav'),
        'val_seen_e': (BASE / 'val/dev_seen/wav').glob('*_enroll.wav'),
        'val_unseen_q': (BASE / 'val/dev_unseen/wav').glob('*_query.wav'),
        'val_unseen_e': (BASE / 'val/dev_unseen/wav').glob('*_enroll.wav'),
        'test_seen_q': (BASE / 'test/eval_seen_unlabeled/wav').glob('*_query.wav'),
        'test_seen_e': (BASE / 'test/eval_seen_unlabeled/wav').glob('*_enroll.wav'),
        'test_unseen_q': (BASE / 'test/eval_unseen_unlabeled/wav').glob('*_query.wav'),
        'test_unseen_e': (BASE / 'test/eval_unseen_unlabeled/wav').glob('*_enroll.wav'),
    }
    by_size = {}  # size -> [(gname, path)]
    for gname, files in groups.items():
        for p in files:
            sz = p.stat().st_size
            by_size.setdefault(sz, []).append((gname, p))
    print(f'大小分组: {len(by_size)} 种大小', flush=True)

    cross_pairs = {
        ('val_seen_q', 'test_seen_q'), ('val_seen_q', 'test_unseen_q'),
        ('val_unseen_q', 'test_unseen_q'), ('val_unseen_q', 'test_seen_q'),
        ('test_seen_q', 'test_unseen_q'),
        ('val_seen_e', 'test_seen_e'), ('val_unseen_e', 'test_unseen_e'),
        ('test_seen_e', 'test_unseen_e'),
        ('val_seen_q', 'test_seen_e'), ('val_unseen_q', 'test_unseen_e'),
    }
    # 2. 对每个大小组, 只对交叉组算 md5 并比较
    candidates = 0
    matches = 0
    for sz, items in by_size.items():
        if len(items) < 2:
            continue
        # 组内按组聚合
        per = {}
        for g, p in items:
            per.setdefault(g, []).append(p)
        gnames = list(per.keys())
        hashes = {}
        for i in range(len(gnames)):
            for j in range(i + 1, len(gnames)):
                g1, g2 = gnames[i], gnames[j]
                if (g1, g2) not in cross_pairs and (g2, g1) not in cross_pairs:
                    continue
                for g in (g1, g2):
                    if g not in hashes:
                        hashes[g] = {md5_of(p) for p in per[g]}
                inter = hashes[g1] & hashes[g2]
                if inter:
                    matches += len(inter)
                    print(f'MATCH {g1} vs {g2}: {len(inter)} 个相同 (size={sz})', flush=True)
                candidates += 1
    print(f'检查了 {candidates} 个大小相同的跨集组, 共 {matches} 个匹配', flush=True)


if __name__ == '__main__':
    main()
