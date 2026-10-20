"""500K train 全局嵌入 Matcher 训练 (音频对 + 文本), 验证 val, 生成 test 提交
用法: python opt/train_500k_matcher.py [--epochs 5] [--bs 1024] [--submit]
输出: opt/out_500k/best_500k.pt + submission/submit_500k.csv
"""
import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BASE = Path(r"D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
OUT = BASE / "opt" / "out_500k"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

CHARS = "abcdefghijklmnopqrstuvwxyz' -"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
MAX_TXT = 24


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def encode_text(txt, max_len=MAX_TXT):
    ids = [CHAR2IDX.get(c, 1) for c in str(txt).lower()][:max_len]
    ids = ids + [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class TextEncoder(nn.Module):
    def __init__(self, emb_dim=64, hidden=128, num_chars=len(CHARS) + 1):
        super().__init__()
        self.emb = nn.Embedding(num_chars, emb_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = self.conv(e)
        w = torch.softmax(h.mean(1, keepdim=True), -1)
        return self.proj((w * h).sum(-1))


class Matcher500K(nn.Module):
    def __init__(self, a_dim=384, hidden=512, text_dim=128):
        super().__init__()
        self.audio_fc = nn.Sequential(
            nn.Linear(a_dim * 4, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
        )
        self.text_enc = TextEncoder()
        self.cls = nn.Sequential(
            nn.Linear(hidden + text_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, e, q, t):
        a = torch.cat([e, q, torch.abs(e - q), e * q], -1)
        af = self.audio_fc(a)
        tf = self.text_enc(t)
        return self.cls(torch.cat([af, tf], -1)).squeeze(-1)


@torch.no_grad()
def predict(model, e, q, t, bs=1024):
    model.eval()
    outs = []
    for i in range(0, len(e), bs):
        eb = torch.from_numpy(e[i:i + bs]).float().to(DEVICE)
        qb = torch.from_numpy(q[i:i + bs]).float().to(DEVICE)
        tb = torch.stack([encode_text(x) for x in t[i:i + bs]]).to(DEVICE)
        outs.append(torch.sigmoid(model(eb, qb, tb)).cpu().numpy())
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.02)
    ap.add_argument("--include-val", action="store_true",
                    help="将 val 10K 加入训练集 (同源录音适配)")
    ap.add_argument("--group-holdout", action="store_true",
                    help="按 query 词分组的 holdout (跨录音评估)")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    set_seed(SEED)

    G = BASE / "output_whisper" / "train_global"
    import glob
    # 支持部分 chunk: 加载所有已完成的 chunk 文件
    e_files = sorted(glob.glob(str(G / "enroll_[0-9][0-9].npy")))
    q_files = sorted(glob.glob(str(G / "query_[0-9][0-9].npy")))
    if not e_files:
        # 全量合并文件
        enroll = np.load(G / "enroll_emb.npy")
        query = np.load(G / "query_emb.npy")
        n_all = len(enroll)
        label_file = G / "labels.npy" if (G / "labels.npy").exists() else G / f"labels_{n_all}.npy"
        text_file = G / "texts.pkl" if (G / "texts.pkl").exists() else G / f"texts_{n_all}.pkl"
        labels = np.load(label_file)
        texts = pickle.load(open(text_file, "rb"))
    else:
        parts = [np.load(f) for f in e_files]
        enroll = np.concatenate(parts, axis=0)
        parts = [np.load(f) for f in q_files]
        query = np.concatenate(parts, axis=0)
        n_all = len(enroll)
        # 找已存在的最大 N label 文件
        import re as _re
        lab_files = sorted((G / "labels_*.npy").exists() and glob.glob(str(G / "labels_*.npy")) or [])
        lab_files += glob.glob(str(G / "labels_*.npy"))
        lab_files = sorted(set(lab_files), key=lambda p: int(_re.search(r'labels_(\d+)\.npy', p).group(1)))
        label_file = lab_files[-1] if lab_files else (G / "labels.npy")
        text_file = G / _re.sub(r'labels_(\d+)\.npy', r'texts_\1.pkl', str(label_file))
        if not Path(text_file).exists():
            text_file = G / "texts.pkl"
        labels = np.load(label_file)
        texts = pickle.load(open(text_file, "rb"))
    # 长度对齐 (嵌入可能比 labels 少, 截取)
    n = min(len(enroll), len(labels))
    enroll, query, labels = enroll[:n], query[:n], labels[:n]
    texts = texts[:n]
    print(f"train: {enroll.shape} {query.shape} pos={labels.mean():.3f} (chunks={len(e_files)})", flush=True)

    # 混合 val 10K (同源录音): 加入训练
    if args.include_val:
        vd = pickle.load(open(BASE / "output_whisper/val_feats.pkl", "rb"))
        ve, vq, vy, vt = [], [], [], []
        for split in ["dev_seen", "dev_unseen"]:
            for pid, it in vd[split].items():
                ve.append(it["enroll_emb"])
                vq.append(it["query_emb"])
                vy.append(it["label"])
                vt.append((it["enroll_txt"], it["query_txt"]))
        ve = np.stack(ve).astype(np.float32)
        vq = np.stack(vq).astype(np.float32)
        vy = np.array(vy, dtype=np.int64)
        vt = np.array(vt, dtype=object)
        # 去重: 与 train 相同的 pid 不重复 (train 是 pair_1..N, val 是独立 pair, 无重叠)
        n_tr = len(enroll)
        enroll = np.concatenate([enroll, ve], axis=0)
        query = np.concatenate([query, vq], axis=0)
        labels = np.concatenate([labels, vy], axis=0)
        texts = np.concatenate([texts, vt], axis=0)
        print(f"混合后: {enroll.shape} (train {n_tr} + val 10K)", flush=True)

    n = len(labels)
    idx = np.arange(n)
    if args.group_holdout:
        # 按 (enroll ∪ query) 词分组: 验证词的任何出现都从训练剔除 (无词级泄漏)
        from collections import defaultdict
        word2idx = defaultdict(list)
        for i in range(n):
            w = str(texts[i][0]).lower().strip()
            word2idx[w].append(i)
            w2 = str(texts[i][1]).lower().strip()
            if w2 != w:
                word2idx[w2].append(i)
        words = sorted(word2idx.keys())
        rng = np.random.RandomState(SEED)
        rng.shuffle(words)
        n_val_words = max(1, int(len(words) * args.holdout * 5))
        val_words = set(words[:n_val_words])
        val_idx = np.array(sorted({i for w in val_words for i in word2idx[w]}))
        tr_mask = np.ones(n, dtype=bool)
        tr_mask[val_idx] = False
        tr_idx = np.arange(n)[tr_mask]
        ho_idx = val_idx
        print(f"[group-holdout(enroll∪query)] {len(words)} 词, 验证 {n_val_words} 词 "
              f"({len(val_idx)} 样本, 训练 {len(tr_idx)} 样本)", flush=True)
    else:
        np.random.shuffle(idx)
        n_ho = int(n * args.holdout)
        ho_idx, tr_idx = idx[:n_ho], idx[n_ho:]

    model = Matcher500K().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n_steps = (len(tr_idx) + args.bs - 1) // args.bs
    from sklearn.metrics import roc_auc_score
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        order = tr_idx.copy()
        np.random.shuffle(order)
        for s in range(n_steps):
            bi = order[s * args.bs:(s + 1) * args.bs]
            eb = torch.from_numpy(enroll[bi]).float().to(DEVICE)
            qb = torch.from_numpy(query[bi]).float().to(DEVICE)
            tb = torch.stack([encode_text(texts[i][0]) for i in bi]).to(DEVICE)
            yb = torch.from_numpy(labels[bi]).float().to(DEVICE)
            logit = model(eb, qb, tb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        # holdout eval
        ho_pred = predict(model, enroll[ho_idx], query[ho_idx], [texts[i][0] for i in ho_idx])
        ho_auc = roc_auc_score(labels[ho_idx], ho_pred)
        print(f"ep{ep} loss={loss.item():.4f} holdout_auc={ho_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if ho_auc > best:
            best = ho_auc
            torch.save(model.state_dict(), OUT / "best_500k.pt")
    print(f"best holdout AUC={best:.4f}", flush=True)

    # val 评估 (参考)
    vd = pickle.load(open(BASE / "output_whisper/val_feats.pkl", "rb"))
    for split in ["dev_seen", "dev_unseen"]:
        pids = sorted(vd[split].keys())
        e = np.stack([vd[split][p]["enroll_emb"] for p in pids])
        q = np.stack([vd[split][p]["query_emb"] for p in pids])
        t = [vd[split][p]["enroll_txt"] for p in pids]
        y = np.array([vd[split][p]["label"] for p in pids])
        pr = predict(model, e, q, t)
        print(f"val {split}: AUC={roc_auc_score(y, pr):.6f} n={len(y)}", flush=True)

    if args.submit:
        for tag, feats_name, type_ in [
            ("inset", "test_inset.pkl", "inset"),
            ("outset", "test_outset.pkl", "outset"),
        ]:
            d = pickle.load(open(BASE / "output_whisper" / feats_name, "rb"))
            pids = sorted(d.keys())
            e = np.stack([d[p]["enroll_emb"] for p in pids])
            q = np.stack([d[p]["query_emb"] for p in pids])
            t = [d[p]["enroll_txt"] for p in pids]
            pr = predict(model, e, q, t)
            preds = dict(zip(pids, pr))
            out_csv = OUT / f"submit_500k_{type_}.csv"
            with open(out_csv, "w", encoding="utf-8") as f:
                for i in range(50000):
                    pid = f"pair_{i+1:06d}"
                    f.write(f"{i},{type_},{preds.get(pid, 0.5):.6f}\n")
            print(f"saved {out_csv}", flush=True)
        # 合并
        def read_csv(tag):
            m = {}
            for line in open(OUT / f"submit_500k_{tag}.csv", encoding="utf-8"):
                if line.startswith("id"): continue
                parts = line.strip().split(",")
                m[int(parts[0])] = parts[2]
            return m
        with open(BASE / "submission/submit_500k.csv", "w", encoding="utf-8", newline="") as f:
            f.write("id,type,prediction\n")
            for tag in ["inset", "outset"]:
                m = read_csv(tag)
                for i in range(50000):
                    f.write(f"{i},{tag},{m[i]}\n")
        print("saved submission/submit_500k.csv", flush=True)


if __name__ == "__main__":
    main()
