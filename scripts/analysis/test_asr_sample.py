"""抽查 test query 的 ASR 转写"""
import csv
import sys
from pathlib import Path

import torch
import whisper
import soundfile as sf

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def main():
    model = whisper.load_model('small')
    opt = whisper.DecodingOptions(language='en', without_timestamps=True, fp16=False)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    which = sys.argv[2] if len(sys.argv) > 2 else 'seen'
    folder = 'eval_seen_unlabeled' if which == 'seen' else 'eval_unseen_unlabeled'
    csv_name = 'eval_seen_enroll_text.csv' if which == 'seen' else 'eval_unseen_enroll_text.csv'
    rows = list(csv.DictReader(open(BASE / 'test' / csv_name, encoding='utf-8')))[:n]
    for r in rows:
        pid = r['id']
        wav = str(BASE / f'test/{folder}/wav/{pid}_query.wav')
        d, sr = sf.read(wav, dtype='float32')
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(torch.from_numpy(d))).unsqueeze(0).to(model.device)
        res = whisper.decode(model, mel, opt)
        texts = res if isinstance(res, list) else [res]
        print(pid, 'enroll=', repr(r['enroll_txt']), 'asr=', repr(texts[0].text), flush=True)


if __name__ == '__main__':
    main()
