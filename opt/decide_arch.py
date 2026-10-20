"""决定性实验: 词级分组 CV 对比 p4(MatcherV4) vs seq(SeqMatcher) 在未见词上的匹配能力
在 val 10K 上按 (enroll ∪ query) 词分组, 训练/验证词不重叠 — 反映 test 未见词难度
"""
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

BASE = Path(r"D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "opt"))
from train_whisper_p4 import MatcherV4, encode_text as encode_text_p4
from train_seqmatcher import SeqMatcher, encode_text as encode_text_seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def load_items():
    d = pickle.load(open(BASE / "output_whisper/val_feats.pkl", "rb"))
    lip = pickle.load(open(BASE / "output_lip/val_lip.pkl", "rb"))
    items = []
    for name in ["dev_seen", "dev_unseen"]:
        for pid, it in d[name].items():
            v = lip[name].get(pid, {})
            items.append({
                "pid": pid,
                "ef": torch.tensor(it["enroll_frames"].astype(np.float32)),
                "qf": torch.tensor(it["query_frames"].astype(np.float32)),
                "eg": torch.tensor(it["enroll_emb"], dtype=torch.float32),
                "qg": torch.tensor(it["query_emb"], dtype=torch.float32),
                "ev": torch.tensor(v.get("enroll_video", np.zeros(256, np.float32)), dtype=torch.float32),
                "qv": torch.tensor(v.get("query_video", np.zeros(256, np.float32)), dtype=torch.float32),
                "word": it["query_txt"].lower().strip(),
                "enroll_word": it["enroll_txt"].lower().strip(),
                "label": float(it["label"]),
            })
    return items


def group_split(items, n_val_words=400, seed=SEED):
    from collections import defaultdict
    word2idx = defaultdict(list)
    for i, it in enumerate(items):
        word2idx[it["word"]].append(i)
        if it["enroll_word"] != it["word"]:
            word2idx[it["enroll_word"]].append(i)
    words = sorted(word2idx.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(words)
    val_words = set(words[:n_val_words])
    val_idx = set(i for w in val_words for i in word2idx[w])
    tr = [it for i, it in enumerate(items) if i not in val_idx]
    va = [it for i, it in enumerate(items) if i in val_idx]
    return tr, va


def make_batch_p4(items, device, bs=256):
    ea = torch.stack([it["eg"] for it in items]).to(device)
    qa = torch.stack([it["qg"] for it in items]).to(device)
    ev = torch.stack([it["ev"] for it in items]).to(device)
    qv = torch.stack([it["qv"] for it in items]).to(device)
    t = torch.stack([encode_text_p4(it["enroll_word"]) for it in items]).to(device)
    y = torch.tensor([it["label"] for it in items], dtype=torch.float32).to(device)
    return ea, qa, ev, qv, t, y


def make_batch_seq(items, device, bs=256):
    ef = torch.stack([it["ef"] for it in items]).to(device)
    qf = torch.stack([it["qf"] for it in items]).to(device)
    eg = torch.stack([it["eg"] for it in items]).to(device)
    qg = torch.stack([it["qg"] for it in items]).to(device)
    t = torch.stack([encode_text_seq(it["enroll_word"]) for it in items]).to(device)
    y = torch.tensor([it["label"] for it in items], dtype=torch.float32).to(device)
    return ef, qf, eg, qg, t, y


def train_eval(model, tr, va, make_batch, epochs=25, bs=256, lr=1e-3):
    import torch.nn.functional as F
    import torch.optim as optim
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = 0.0
    for ep in range(epochs):
        model.train()
        idx = list(range(len(tr)))
        random.shuffle(idx)
        for i in range(0, len(idx), bs):
            bi = [tr[j] for j in idx[i:i + bs]]
            inp = make_batch(bi, DEVICE)
            y = inp[-1]
            logit = model(*inp[:-1])
            if logit.dim() > 1:
                logit = logit.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for i in range(0, len(va), bs):
                inp = make_batch(va[i:i + bs], DEVICE)
                logit = model(*inp[:-1])
                if logit.dim() > 1:
                    logit = logit.squeeze(-1)
                preds.extend(torch.sigmoid(logit).cpu().flatten().tolist())
                ys.extend(inp[-1].cpu().tolist())
            auc = roc_auc_score(ys, preds)
        if auc > best:
            best = auc
    return best


def predict_all(model, items, make_batch, bs=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            inp = make_batch(items[i:i + bs], DEVICE)
            logit = model(*inp[:-1])
            if logit.dim() > 1:
                logit = logit.squeeze(-1)
            preds.extend(torch.sigmoid(logit).cpu().flatten().tolist())
    return np.array(preds)


def train_predict(model, tr, va, make_batch, epochs=25, bs=256, lr=1e-3):
    import torch.nn.functional as F
    import torch.optim as optim
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        idx = list(range(len(tr)))
        random.shuffle(idx)
        for i in range(0, len(idx), bs):
            bi = [tr[j] for j in idx[i:i + bs]]
            inp = make_batch(bi, DEVICE)
            y = inp[-1]
            logit = model(*inp[:-1])
            if logit.dim() > 1:
                logit = logit.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return predict_all(model, va, make_batch, bs)


def main():
    items = load_items()
    print(f"val 10K items, 词数: {len(set(it['word'] for it in items))}", flush=True)
    tr, va = group_split(items, n_val_words=300)
    print(f"训练 {len(tr)} 验证 {len(va)} (验证词 300)", flush=True)
    ys = np.array([it['label'] for it in va])

    print("=== MatcherV4 ===", flush=True)
    p4_pred = train_predict(MatcherV4(), tr, va, make_batch_p4, epochs=25)
    print(f"  p4 未见词 AUC = {roc_auc_score(ys, p4_pred):.4f}", flush=True)

    print("=== SeqMatcher ===", flush=True)
    seq_pred = train_predict(SeqMatcher(), tr, va, make_batch_seq, epochs=25)
    print(f"  seq 未见词 AUC = {roc_auc_score(ys, seq_pred):.4f}", flush=True)

    # 融合评估 (rank 平均)
    from scipy.stats import rankdata
    p4_r = rankdata(p4_pred) / len(p4_pred)
    seq_r = rankdata(seq_pred) / len(seq_pred)
    for w in [0.3, 0.5, 0.7]:
        fused = w * p4_r + (1 - w) * seq_r
        print(f"  融合 w_p4={w}: AUC={roc_auc_score(ys, fused):.4f}", flush=True)
    print(f"\n>>> 结论: p4 单独 vs 融合, 选择未见词 AUC 最高者", flush=True)


if __name__ == "__main__":
    main()
