import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
提取 train 500K 对的 whisper 特征 -> output_whisper/train_feats.npz
用法: python extract_train_feats.py
输出: enroll_emb(500000,384) query_emb(500000,384) labels(500000,) texts(500000,2)
"""
import os, sys, time, pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from tqdm import tqdm

from whisper_encoder import WhisperEncoder, SR

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
TRAIN = BASE / "train"
CKPT = BASE / "辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt"
OUT = BASE / "output_whisper" / "train_feats.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 256          # 每批 pair 数 (每个 pair 2 条音频)
ENROLL_MAX = 1.5
QUERY_MAX = 2.0
N_PAIRS = 500000


def load_audio(path, max_sec):
    d, sr = sf.read(str(path), dtype='float32')
    if d.ndim > 1:
        d = d.mean(axis=1)
    a = torch.from_numpy(d).float()
    if sr != SR:
        a = torchaudio_force_resample(a, sr)
    max_s = int(max_sec * SR)
    if a.shape[0] > max_s:
        a = a[:max_s]
    return a


def torchaudio_force_resample(a, sr):
    import torchaudio
    return torchaudio.functional.resample(a, sr, SR)


@torch.no_grad()
def embed_paths(enc, paths, max_sec, desc):
    """paths: list[str] -> np.ndarray (N, 384)"""
    out = []
    for i in tqdm(range(0, len(paths), BATCH), desc=desc, leave=False):
        chunk = paths[i:i + BATCH]
        waves, valid_lens = [], []
        for p in chunk:
            w = load_audio(p, max_sec)
            waves.append(w)
            valid_lens.append(w.shape[0])
        maxl = max(w.shape[0] for w in waves)
        batch = torch.stack([F.pad(w, (0, maxl - w.shape[0])) for w in waves]).to(DEVICE)
        vl = torch.tensor(valid_lens, dtype=torch.long)
        emb, _ = enc.embed_with_frames(batch, valid_lens=vl, n_frames=24)
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0)


def read_label(pid):
    parts = open(TRAIN / "label" / f"{pid}.txt", encoding='utf-8').read().strip().split('\t')
    label = int(parts[2].replace('label:', ''))
    enroll_txt = parts[0].replace('enrollment:', '')
    query_txt = parts[1].replace('query:', '')
    return label, enroll_txt, query_txt


def main():
    t0 = time.time()
    enc = WhisperEncoder(CKPT).to(DEVICE)
    enc.eval()
    print(f"Device: {DEVICE}")

    pids = [f"pair_{i:06d}" for i in range(1, N_PAIRS + 1)]

    enroll_paths = [str(TRAIN / "wav" / f"{p}_enroll.wav") for p in pids]
    query_paths = [str(TRAIN / "wav" / f"{p}_query.wav") for p in pids]

    print("[1] enroll 特征提取...")
    enroll_emb = embed_paths(enc, enroll_paths, ENROLL_MAX, "enroll")
    print(f"  enroll_emb {enroll_emb.shape} ({time.time()-t0:.0f}s)")

    print("[2] query 特征提取...")
    query_emb = embed_paths(enc, query_paths, QUERY_MAX, "query")
    print(f"  query_emb {query_emb.shape} ({time.time()-t0:.0f}s)")

    print("[3] 读 label/文本...")
    labels = np.zeros(N_PAIRS, dtype=np.int64)
    texts = np.empty((N_PAIRS, 2), dtype=object)
    for i, p in enumerate(tqdm(pids, leave=False)):
        lab, et, qt = read_label(p)
        labels[i] = lab
        texts[i] = (et, qt)
    print(f"  pos rate: {labels.mean():.3f}")

    np.savez_compressed(OUT, enroll_emb=enroll_emb, query_emb=query_emb,
                        labels=labels, texts=texts)
    print(f"saved {OUT} ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
