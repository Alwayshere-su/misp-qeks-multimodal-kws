"""分析 val: ASR 精确匹配的正样本中, 声学分数低的占比 (判断硬加分收益)"""
import pickle
import re
from pathlib import Path

import numpy as np

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9' ]", '', (s or '').lower()).strip()


def exact_match(enroll, text):
    e = norm(enroll)
    q = norm(text)
    if not e or not q:
        return 0.0
    return 1.0 if (e == q or e in q or q in e) else 0.0


def main():
    asr = pickle.load(open(BASE / 'output_whisper/val_asr_small.pkl', 'rb'))
    aco = pickle.load(open(BASE / 'output_p4/s2u_scores.pkl', 'rb'))  # 无泄漏声学
    for split in ['dev_seen', 'dev_unseen']:
        labels = {}
        for f in (BASE / f'val/{split}/label').glob('*.txt'):
            parts = f.read_text(encoding='utf-8').strip().split('\t')
            labels[f.stem] = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in parts}
        n_pos = sum(1 for d in labels.values() if d['label'] == '1')
        # ASR 精确匹配的正样本中, 声学分数分布
        hit_low_aco = hit_total = 0
        miss_low_aco = miss_total = 0
        for pid, d in labels.items():
            s = exact_match(d['enrollment'], asr[f'{split}/{pid}'])
            ac = aco[split][pid]
            is_pos = d['label'] == '1'
            if s >= 0.999:
                hit_total += 1
                if ac < 0.5:
                    hit_low_aco += 1
            else:
                if is_pos:
                    miss_total += 1
                    if ac < 0.5:
                        miss_low_aco += 1
        print(f'=== {split} ===', flush=True)
        print(f'  ASR命中: {hit_total} 个, 其中声学<0.5: {hit_low_aco} ({hit_low_aco / max(hit_total,1):.1%})', flush=True)
        print(f'  ASR未命中但正样本: {miss_total} 个, 其中声学<0.5: {miss_low_aco} ({miss_low_aco / max(miss_total,1):.1%})', flush=True)
        # 声学分数 <0.5 的正样本: 有多少能被 ASR 救回
        pos_low_aco = sum(1 for pid, d in labels.items() if d['label'] == '1' and aco[split][pid] < 0.5)
        pos_low_aco_asr_hit = sum(1 for pid, d in labels.items()
                                  if d['label'] == '1' and aco[split][pid] < 0.5
                                  and exact_match(d['enrollment'], asr[f'{split}/{pid}']) >= 0.999)
        print(f'  声学<0.5的正样本: {pos_low_aco}, 其中ASR命中: {pos_low_aco_asr_hit} '
              f'({pos_low_aco_asr_hit / max(pos_low_aco,1):.1%})', flush=True)


if __name__ == '__main__':
    main()
