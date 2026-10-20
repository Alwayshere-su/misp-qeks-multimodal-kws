"""温度采样实验: 多次解码能否救回 ASR 失败的正样本"""
import re
from pathlib import Path

import torch
import whisper
import soundfile as sf

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9' ]", '', (s or '').lower()).strip()


def main():
    model = whisper.load_model('small')
    labels = {}
    for f in (BASE / 'val/dev_unseen/label').glob('*.txt'):
        parts = f.read_text(encoding='utf-8').strip().split('\t')
        labels[f.stem] = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in parts}
    pos = [(pid, d) for pid, d in labels.items() if d['label'] == '1'][:80]

    temps = [0.0, 0.2, 0.5, 0.8, 1.0]
    opts = [whisper.DecodingOptions(language='en', without_timestamps=True,
                                    fp16=False, temperature=t) for t in temps]
    hit_single = hit_multi = total = 0
    examples = []
    for pid, d in pos:
        wav = str(BASE / f'val/dev_unseen/wav/{pid}_query.wav')
        a, sr = sf.read(wav, dtype='float32')
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(torch.from_numpy(a))).to(model.device)
        e = norm(d['enrollment'])
        texts = []
        for o in opts:
            r = whisper.decode(model, mel, o)
            texts.append(norm(r.text))
        single_hit = (e == texts[0] or e in texts[0] or texts[0] in e)
        multi_hit = any(e == t or e in t or t in e for t in texts)
        hit_single += single_hit
        hit_multi += multi_hit
        total += 1
        if single_hit and not multi_hit:
            pass
        if not single_hit:
            examples.append((pid, e, texts))
    print(f'n={total} 单次贪心={hit_single} ({hit_single/total:.0%})  '
          f'温度采样任一命中={hit_multi} ({hit_multi/total:.0%})')
    print('--- 未命中样本的多采样输出 ---')
    for pid, e, texts in examples[:25]:
        print(f'{pid} enroll={e!r}')
        for t, txt in zip(temps, texts):
            print(f'   t={t}: {txt!r}')


if __name__ == '__main__':
    main()
