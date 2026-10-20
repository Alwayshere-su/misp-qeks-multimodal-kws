"""评估 ASR 在 test 上的匹配分数分布 (判断 ASR 融合价值)"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import whisper
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from asr_pipeline import match_score

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def main():
    model = whisper.load_model('small')
    opt = whisper.DecodingOptions(language='en', without_timestamps=True, fp16=False)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    for tag, folder, csv_name in [
        ('test_seen', 'eval_seen_unlabeled', 'eval_seen_enroll_text.csv'),
        ('test_unseen', 'eval_unseen_unlabeled', 'eval_unseen_enroll_text.csv'),
    ]:
        rows = list(csv.DictReader(open(BASE / 'test' / csv_name, encoding='utf-8')))[:n]
        scores = []
        for r in rows:
            wav = str(BASE / f'test/{folder}/wav/{r["id"]}_query.wav')
            d, sr = sf.read(wav, dtype='float32')
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(torch.from_numpy(d))).unsqueeze(0).to(model.device)
            res = whisper.decode(model, mel, opt)
            texts = res if isinstance(res, list) else [res]
            scores.append(match_score(r['enroll_txt'], texts[0].text, 'edit'))
        s = np.array(scores)
        print(f'{tag}: n={len(s)} mean={s.mean():.3f} '
              f'gt0.8={(s > 0.8).mean():.3f} ge0.999={(s >= 0.999).mean():.3f} '
              f'gt0.5={(s > 0.5).mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
