"""Extract train 500K whisper global embeddings (chunked, checkpointed).
Output: output_whisper/train_global/{enroll_emb.npy, query_emb.npy, labels.npy, texts.pkl}
"""
import sys, time, pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
TRAIN = BASE / "train"
CKPT = BASE / "辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt"
OUT_DIR = BASE / "output_whisper" / "train_global"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE))
from whisper_encoder import WhisperEncoder, SR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 256
ENROLL_MAX = 1.5
QUERY_MAX = 2.0
N_PAIRS = 500000
CHUNK = 50000


def load_audio(path, max_sec):
    d, sr = sf.read(str(path), dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    a = torch.from_numpy(d).float()
    if sr != SR:
        import torchaudio
        a = torchaudio.functional.resample(a, sr, SR)
    max_s = int(max_sec * SR)
    if a.shape[0] > max_s:
        a = a[:max_s]
    return a


@torch.no_grad()
def embed_paths(enc, paths, max_sec, desc):
    out = []
    for i in range(0, len(paths), BATCH):
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
        if (i // BATCH) % 20 == 0:
            print(f"    {desc}: {i + len(chunk)}/{len(paths)} ({time.time()-_t0:.0f}s)", flush=True)
    return np.concatenate(out, axis=0)


def read_label(pid):
    parts = open(TRAIN / "label" / f"{pid}.txt", encoding="utf-8").read().strip().split("\t")
    label = int(parts[2].replace("label:", ""))
    enroll_txt = parts[0].replace("enrollment:", "")
    query_txt = parts[1].replace("query:", "")
    return label, enroll_txt, query_txt


_t0 = time.time()
enc = WhisperEncoder(CKPT).to(DEVICE)
enc.eval()
print(f"Device: {DEVICE}", flush=True)

pids = [f"pair_{i:06d}" for i in range(1, N_PAIRS + 1)]

# labels/texts 由 opt/prep_train_labels.py 单独生成 (txt 小文件 IO 慢, 后台补); 嵌入不依赖 label
if (OUT_DIR / "labels.npy").exists() and (OUT_DIR / "texts.pkl").exists():
    print("[labels/texts] 已存在", flush=True)
else:
    print("[labels/texts] 暂缺, 嵌入照常进行 (由 prep_train_labels.py 补)", flush=True)

# chunked embedding extraction with checkpointing
for c in range(0, N_PAIRS, CHUNK):
    cid = c // CHUNK
    if (OUT_DIR / f"enroll_{cid:02d}.npy").exists():
        print(f"chunk {cid} exists, skip", flush=True)
        continue
    e_paths = [str(TRAIN / "wav" / f"{p}_enroll.wav") for p in pids[c:c + CHUNK]]
    q_paths = [str(TRAIN / "wav" / f"{p}_query.wav") for p in pids[c:c + CHUNK]]
    print(f"[chunk {cid}] enroll...", flush=True)
    e_emb = embed_paths(enc, e_paths, ENROLL_MAX, f"enroll{cid}")
    print(f"[chunk {cid}] query...", flush=True)
    q_emb = embed_paths(enc, q_paths, QUERY_MAX, f"query{cid}")
    np.save(OUT_DIR / f"enroll_{cid:02d}.npy", e_emb)
    np.save(OUT_DIR / f"query_{cid:02d}.npy", q_emb)
    print(f"[chunk {cid}] saved ({time.time()-_t0:.0f}s)", flush=True)

# merge
print("[merge] ...", flush=True)
enroll = np.concatenate([np.load(OUT_DIR / f"enroll_{c:02d}.npy") for c in range(N_PAIRS // CHUNK)], axis=0)
query = np.concatenate([np.load(OUT_DIR / f"query_{c:02d}.npy") for c in range(N_PAIRS // CHUNK)], axis=0)
np.save(OUT_DIR / "enroll_emb.npy", enroll)
np.save(OUT_DIR / "query_emb.npy", query)
print(f"[DONE] {enroll.shape} {query.shape} ({time.time()-_t0:.0f}s)", flush=True)
