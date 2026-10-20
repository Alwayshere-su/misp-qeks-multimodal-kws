"""分析: ASR 精确匹配是否为强正样本信号 (val 上验证) — 纯 Python 无 torch"""
import pickle
import re
from pathlib import Path

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
    for split in ['dev_seen', 'dev_unseen']:
        labels = {}
        for f in (BASE / f'val/{split}/label').glob('*.txt'):
            parts = f.read_text(encoding='utf-8').strip().split('\t')
            labels[f.stem] = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in parts}
        exact_pos = exact_neg = 0
        n_pos = sum(1 for d in labels.values() if d['label'] == '1')
        for pid, d in labels.items():
            s = exact_match(d['enrollment'], asr[f'{split}/{pid}'])
            if s >= 0.999:
                if d['label'] == '1':
                    exact_pos += 1
                else:
                    exact_neg += 1
        print(f'{split}: ASR精确匹配 正={exact_pos} 负={exact_neg} | 总正={n_pos} '
              f'| 命中正样本={exact_pos}/{n_pos} | 误报={exact_neg}', flush=True)
        if exact_pos + exact_neg > 0:
            print(f'  精确匹配中正样本占比: {exact_pos / (exact_pos + exact_neg):.4f}', flush=True)


if __name__ == '__main__':
    main()
