"""分析 ASR 失败样本: 正样本 (label=1) 中 ASR 转写与 enroll 的差距"""
import re
import sys
from pathlib import Path

import torch
import whisper
import soundfile as sf

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9' ]", '', (s or '').lower()).strip()


def edit_dist(a, b):
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[lb]


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else 'dev_unseen'
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    model_name = sys.argv[3] if len(sys.argv) > 3 else 'small'
    model = whisper.load_model(model_name)
    opt = whisper.DecodingOptions(language='en', without_timestamps=True, fp16=False)

    labels = {}
    for f in (BASE / f'val/{split}/label').glob('*.txt'):
        parts = f.read_text(encoding='utf-8').strip().split('\t')
        labels[f.stem] = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in parts}

    pos = [(pid, d) for pid, d in labels.items() if d['label'] == '1'][:limit]
    hit, miss = [], []
    for pid, d in pos:
        wav = str(BASE / f'val/{split}/wav/{pid}_query.wav')
        a, sr = sf.read(wav, dtype='float32')
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(torch.from_numpy(a))).to(model.device)
        res = whisper.decode(model, mel, opt)
        e, q = norm(d['enrollment']), norm(res.text)
        sim = 1.0 - edit_dist(e, q) / max(len(e), len(q), 1)
        (hit if sim >= 0.8 else miss).append((pid, d['enrollment'], res.text, round(sim, 2)))

    print(f'[{model_name} {split}] hit={len(hit)} miss={len(miss)}')
    print('--- 失败样本 (enroll | asr | sim) ---')
    for pid, e, a, s in miss[:50]:
        print(f'{pid} enroll={e!r} asr={a!r} sim={s}')


if __name__ == '__main__':
    main()
