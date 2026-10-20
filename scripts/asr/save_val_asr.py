import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""val 全量 ASR 转写并缓存 (供打分函数迭代复用)
用法: python save_val_asr.py --model small --out output_whisper/val_asr_small.pkl
"""
import argparse
import pickle
import re
import time
from pathlib import Path

import soundfile as sf
import torch
import whisper

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9' ]", '', (s or '').lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='small')
    ap.add_argument('--out', default='output_whisper/val_asr_small.pkl')
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()

    model = whisper.load_model(args.model)
    opt = whisper.DecodingOptions(language='en', without_timestamps=True, fp16=False)

    cache = {}
    out_path = BASE / args.out
    if out_path.exists():
        cache = pickle.load(open(out_path, 'rb'))
        print(f'缓存已有 {len(cache)} 条', flush=True)

    for split in ['dev_seen', 'dev_unseen']:
        labels = {}
        for f in (BASE / f'val/{split}/label').glob('*.txt'):
            parts = f.read_text(encoding='utf-8').strip().split('\t')
            labels[f.stem] = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in parts}
        pids = sorted(labels.keys())
        todo = [pid for pid in pids if f'{split}/{pid}' not in cache]
        print(f'[{split}] 待转写 {len(todo)}/{len(pids)}', flush=True)
        t0 = time.time()
        done = 0
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            try:
                mels = []
                for pid in chunk:
                    wav = str(BASE / f'val/{split}/wav/{pid}_query.wav')
                    d, sr = sf.read(wav, dtype='float32')
                    a = whisper.pad_or_trim(torch.from_numpy(d))
                    mels.append(whisper.log_mel_spectrogram(a))
                mel_batch = torch.stack(mels).to(model.device)
                results = whisper.decode(model, mel_batch, opt)
                if not isinstance(results, list):
                    results = [results]
                for pid, r in zip(chunk, results):
                    cache[f'{split}/{pid}'] = r.text
            except Exception as e:
                print(f'  [warn] 批次 {i} 失败: {type(e).__name__} {str(e)[:60]}, 逐样本重试', flush=True)
                for pid in chunk:
                    try:
                        wav = str(BASE / f'val/{split}/wav/{pid}_query.wav')
                        d, sr = sf.read(wav, dtype='float32')
                        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(torch.from_numpy(d))).unsqueeze(0).to(model.device)
                        r = whisper.decode(model, mel, opt)
                        cache[f'{split}/{pid}'] = r.text
                    except Exception as e2:
                        cache[f'{split}/{pid}'] = ''
                        print(f'    {pid} 失败: {type(e2).__name__} {str(e2)[:50]}', flush=True)
            done += len(chunk)
            if (done // 200) % 5 == 0 or done == len(todo):
                pickle.dump(cache, open(out_path, 'wb'))
                print(f'  {done}/{len(todo)} 用时 {time.time() - t0:.0f}s 缓存{len(cache)}', flush=True)
        pickle.dump(cache, open(out_path, 'wb'))
        print(f'[{split}] done, 共 {len(cache)} 条', flush=True)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
