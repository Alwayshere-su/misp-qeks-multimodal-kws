"""独立处理 500K train label 阶段 (带进度+异常处理)"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np

BASE = Path(r"D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
TRAIN = BASE / "train"
OUT_DIR = BASE / "output_whisper" / "train_global"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_PAIRS = 500000


def read_label(pid):
    parts = open(TRAIN / "label" / f"{pid}.txt", encoding="utf-8").read().strip().split("\t")
    label = int(parts[2].replace("label:", ""))
    enroll_txt = parts[0].replace("enrollment:", "")
    query_txt = parts[1].replace("query:", "")
    return label, enroll_txt, query_txt


t0 = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=N_PAIRS, help="只处理前 N 个 pair")
args = ap.parse_args()
n_pairs = args.limit
labels = np.zeros(n_pairs, dtype=np.int64)
texts = np.empty((n_pairs, 2), dtype=object)
errs = 0
for i in range(1, n_pairs + 1):
    pid = f"pair_{i:06d}"
    try:
        lab, et, qt = read_label(pid)
        labels[i - 1] = lab
        texts[i - 1] = (et, qt)
    except Exception as e:
        errs += 1
        labels[i - 1] = 0
        texts[i - 1] = ("", "")
        if errs <= 5:
            print(f"ERR {pid}: {type(e).__name__} {e}", flush=True)
    if i % 50000 == 0:
        print(f"  {i}/{n_pairs} errs={errs} ({time.time()-t0:.0f}s)", flush=True)
print(f"label 阶段完成: errs={errs} ({time.time()-t0:.0f}s)", flush=True)
np.save(OUT_DIR / f"labels_{n_pairs}.npy", labels)
with open(OUT_DIR / f"texts_{n_pairs}.pkl", "wb") as f:
    pickle.dump(texts, f)
print(f"labels_{n_pairs}.npy + texts_{n_pairs}.pkl 已保存 ({time.time()-t0:.0f}s)", flush=True)
