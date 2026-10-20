"""提取 test 全量 whisper base 嵌入 (enroll+query) -> 保存 pkl
之后用于 base 多模板 + tiny 融合
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')
sys.path.insert(0, str(BASE))
from opt.base_small_mt import embed_whisper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_split(model, d, folder, out_path, batch=32):
    pids = sorted(d.keys())
    cache = {}
    if out_path.exists():
        cache = pickle.load(open(out_path, 'rb'))
        print(f'缓存 {len(cache)}/{len(pids)}', flush=True)
    todo = [p for p in pids if p not in cache]
    print(f'待提取 {len(todo)}', flush=True)
    t0 = time.time()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        e_b, q_b, e_vl, q_vl = [], [], [], []
        for pid in chunk:
            for kind, lst, vl in [('enroll', e_b, e_vl), ('query', q_b, q_vl)]:
                wav = str(BASE / f'test/{folder}/wav/{pid}_{kind}.wav')
                a, sr = sf.read(wav, dtype='float32')
                a = torch.from_numpy(a).float()
                lst.append(a)
                vl.append(a.shape[0])
        e_embs = embed_whisper(model, e_b, e_vl)
        q_embs = embed_whisper(model, q_b, q_vl)
        for j, pid in enumerate(chunk):
            cache[pid] = (e_embs[j], q_embs[j])
        if (i // batch) % 20 == 0:
            pickle.dump(cache, open(out_path, 'wb'))
            print(f'  {i + len(chunk)}/{len(todo)} ({time.time()-t0:.0f}s)', flush=True)
    pickle.dump(cache, open(out_path, 'wb'))
    print(f'done {len(cache)} ({time.time()-t0:.0f}s)', flush=True)


def main():
    model = whisper.load_model('base')
    model.eval()
    print(f'base loaded, device={model.device}', flush=True)
    tin = pickle.load(open(BASE / 'output_whisper/test_inset.pkl', 'rb'))
    tout = pickle.load(open(BASE / 'output_whisper/test_outset.pkl', 'rb'))
    extract_split(model, tin, 'eval_seen_unlabeled', BASE / 'output_whisper/test_base_inset.pkl')
    extract_split(model, tout, 'eval_unseen_unlabeled', BASE / 'output_whisper/test_base_outset.pkl')


if __name__ == '__main__':
    main()
