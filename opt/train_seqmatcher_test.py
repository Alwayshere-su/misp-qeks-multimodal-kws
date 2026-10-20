"""SeqMatcher: val 全量 5 折训练 -> test 集成提交 (帧级匹配)
用法: python opt/train_seqmatcher_test.py
输出: submission/submit_seq_foldavg.csv
"""
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(r"D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "opt"))
from train_seqmatcher import SeqMatcher, load_items, encode_text, DEVICE, SEED

OUT = BASE / "opt" / "out_seq"


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_test(name):
    d = pickle.load(open(BASE / "output_whisper" / name, "rb"))
    items = []
    for pid, it in d.items():
        items.append({
            "pid": pid,
            "ef": torch.tensor(it["enroll_frames"].astype(np.float32)),
            "qf": torch.tensor(it["query_frames"].astype(np.float32)),
            "eg": torch.tensor(it["enroll_emb"], dtype=torch.float32),
            "qg": torch.tensor(it["query_emb"], dtype=torch.float32),
            "text": encode_text(it["enroll_txt"]),
            "label": 0.0,
        })
    return items


@torch.no_grad()
def predict(model, items, bs=512):
    model.eval()
    outs = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        ef = torch.stack([it["ef"] for it in chunk]).to(DEVICE)
        qf = torch.stack([it["qf"] for it in chunk]).to(DEVICE)
        eg = torch.stack([it["eg"] for it in chunk]).to(DEVICE)
        qg = torch.stack([it["qg"] for it in chunk]).to(DEVICE)
        t = torch.stack([it["text"] for it in chunk]).to(DEVICE)
        logit = model(ef, qf, eg, qg, t)
        outs.extend(torch.sigmoid(logit).cpu().flatten().tolist())
    return dict(zip([it["pid"] for it in items], outs))


def main():
    set_seed(SEED)
    test_sets = {
        "inset": load_test("test_inset.pkl"),
        "outset": load_test("test_outset.pkl"),
    }
    acc = {k: {} for k in test_sets}
    for fold in range(5):
        ckpt = OUT / f"best_fold{fold}.pt"
        if not ckpt.exists():
            print(f"缺 {ckpt}, 跳过", flush=True)
            continue
        model = SeqMatcher().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.eval()
        for tag, items in test_sets.items():
            preds = predict(model, items)
            for pid, pr in preds.items():
                acc[tag].setdefault(pid, []).append(pr)
        print(f"fold{fold} 完成", flush=True)
    with open(BASE / "submission/submit_seq_foldavg.csv", "w", encoding="utf-8", newline="") as f:
        f.write("id,type,prediction\n")
        for tag, order in [("inset", "inset"), ("outset", "outset")]:
            for i in range(50000):
                pid = f"pair_{i+1:06d}"
                if pid in acc[tag]:
                    pr = float(np.mean(acc[tag][pid]))
                else:
                    pr = 0.5
                f.write(f"{i},{order},{pr:.6f}\n")
    print("saved submission/submit_seq_foldavg.csv", flush=True)


if __name__ == "__main__":
    main()
