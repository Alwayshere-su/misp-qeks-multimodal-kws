import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Whisper 特征提取 — val(10K) + test(100K) 的 enroll/query 音频嵌入
=================================================================
输出缓存:
  output_whisper/val_feats.pkl      {pid: {enroll_emb, query_emb, label, enroll_txt, query_txt}}
  output_whisper/test_inset.pkl     {pid: {enroll_emb, query_emb, enroll_txt}}
  output_whisper/test_outset.pkl    {pid: {enroll_emb, query_emb, enroll_txt}}
"""
import os, sys, pickle, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from tqdm import tqdm

from whisper_encoder import WhisperEncoder, SR

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
VAL = BASE / "val"
TEST = BASE / "test"
CKPT = BASE / "辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt"
OUT = BASE / "output_whisper"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 256
ENROLL_MAX = 1.5   # 秒
QUERY_MAX = 2.0    # 秒


def load_audio(path, max_sec=None):
    """返回 (audio_tensor, valid_len)"""
    d, sr = sf.read(str(path), dtype='float32')
    if d.ndim > 1:
        d = d.mean(axis=1)
    a = torch.from_numpy(d).float()
    if sr != SR:
        a = torchaudio_force_resample(a, sr)
    valid = a.shape[0]
    if max_sec is not None:
        max_s = int(max_sec * SR)
        if valid > max_s:
            a = a[:max_s]
            valid = max_s
    return a, valid


def torchaudio_force_resample(a, sr):
    import torchaudio
    return torchaudio.functional.resample(a, sr, SR)


@torch.no_grad()
def embed_batch(enc, paths, max_sec, desc):
    """批量提取嵌入 (变长 + 有效帧 mask 池化), 返回 {path: (emb(384,), frames(24,384) fp16)}"""
    out = {}
    for i in tqdm(range(0, len(paths), BATCH), desc=desc, leave=False):
        chunk = paths[i:i + BATCH]
        waves, valid_lens = [], []
        for p in chunk:
            w, v = load_audio(p, max_sec)
            waves.append(w)
            valid_lens.append(v)
        maxl = max(w.shape[0] for w in waves)
        batch = torch.stack([F.pad(w, (0, maxl - w.shape[0])) for w in waves]).to(DEVICE)
        vl = torch.tensor(valid_lens, dtype=torch.long)
        emb, frames = enc.embed_with_frames(batch, valid_lens=vl, n_frames=24)  # (B,384) (B,24,384)
        for p, e, fr in zip(chunk, emb.cpu().numpy(), frames.cpu().numpy()):
            out[str(p)] = (e, fr.astype(np.float16))
    return out


def read_label(pid, split_dir):
    """返回 (label, enroll_txt, query_txt)"""
    lp = Path(split_dir) / "label" / f"{pid}.txt"
    parts = open(lp, encoding='utf-8').read().strip().split('\t')
    label = int(parts[2].replace('label:', ''))
    enroll_txt = parts[0].replace('enrollment:', '')
    query_txt = parts[1].replace('query:', '')
    return label, enroll_txt, query_txt


def extract_split(enc, split_dir, split_name, is_test=False, text_map=None):
    """split_dir: val/dev_seen 或 test/eval_seen_unlabeled"""
    d = Path(split_dir)
    wav_dir = d / "wav"
    enroll_paths = sorted(wav_dir.glob("pair_*_enroll.wav"))
    query_paths = sorted(wav_dir.glob("pair_*_query.wav"))
    print(f"  {split_name}: {len(enroll_paths)} pairs")

    e_embs = embed_batch(enc, enroll_paths, ENROLL_MAX, f"  {split_name} enroll")
    q_embs = embed_batch(enc, query_paths, QUERY_MAX, f"  {split_name} query")

    data = {}
    for ep in enroll_paths:
        pid = ep.stem.replace('_enroll', '')
        qp = wav_dir / f"{pid}_query.wav"
        e_emb, e_fr = e_embs[str(ep)]
        q_emb, q_fr = q_embs[str(qp)]
        item = {
            'enroll_emb': e_emb,
            'enroll_frames': e_fr,
            'query_emb': q_emb,
            'query_frames': q_fr,
        }
        if is_test:
            item['enroll_txt'] = (text_map or {}).get(pid, '')
        else:
            label, et, qt = read_label(pid, d)
            item['label'] = label
            item['enroll_txt'] = et
            item['query_txt'] = qt
        data[pid] = item
    return data


def load_test_texts():
    """读取 test 的 enroll 文本: {inset: {pid: txt}, outset: {...}}"""
    res = {}
    for key, path in [('inset', TEST / "eval_seen_enroll_text.csv"),
                      ('outset', TEST / "eval_unseen_enroll_text.csv")]:
        m = {}
        with open(path, encoding='utf-8') as f:
            next(f)
            for line in f:
                pid, txt = line.strip().split(',', 1)
                m[pid] = txt
        res[key] = m
    return res


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-only', action='store_true')
    args = parser.parse_args()

    t0 = time.time()
    enc = WhisperEncoder(CKPT).to(DEVICE)
    enc.eval()
    print(f"Device: {DEVICE}")

    # ---- val ----
    print("\n[1] val 特征提取...")
    val_data = {}
    for name in ['dev_seen', 'dev_unseen']:
        val_data[name] = extract_split(enc, VAL / name, f"val/{name}")
    with open(OUT / "val_feats.pkl", 'wb') as f:
        pickle.dump(val_data, f)
    print(f"  saved val_feats.pkl ({time.time()-t0:.0f}s)")

    if args.val_only:
        print("\n[val-only] done")
        return

    # ---- test ----
    print("\n[2] test 特征提取...")
    text_maps = load_test_texts()
    test_data = {}
    for key, name in [('inset', 'eval_seen_unlabeled'), ('outset', 'eval_unseen_unlabeled')]:
        test_data[key] = extract_split(enc, TEST / name, f"test/{key}", is_test=True,
                                       text_map=text_maps[key])
        with open(OUT / f"test_{key}.pkl", 'wb') as f:
            pickle.dump(test_data[key], f)
        print(f"  saved test_{key}.pkl ({time.time()-t0:.0f}s)")

    print(f"\n[DONE] total {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
