"""跨录音泛化实验: train 前80%训练, 后20%验证 (同词不同录音 — test_seen 的核心挑战)
对比: 基线 / 噪声增强 / 对比损失
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score

BASE = Path(r"D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
G = BASE / "output_whisper" / "train_global"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

CHARS = "abcdefghijklmnopqrstuvwxyz' -"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
MAX_TXT = 24


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


class Matcher(nn.Module):
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


def load_data():
    import glob
    e_files = sorted(glob.glob(str(G / "enroll_[0-9][0-9].npy")))
    q_files = sorted(glob.glob(str(G / "query_[0-9][0-9].npy")))
    enroll = np.concatenate([np.load(f) for f in e_files], axis=0)
    query = np.concatenate([np.load(f) for f in q_files], axis=0)
    labels = np.load(G / "labels_500000.npy")
    texts = pickle.load(open(G / "texts_500000.pkl", "rb"))
    n = min(len(enroll), len(labels))
    return enroll[:n], query[:n], labels[:n], texts[:n]


@torch.no_grad()
def predict(model, e, q, t, bs=2048):
    model.eval()
    outs = []
    for i in range(0, len(e), bs):
        eb = torch.from_numpy(e[i:i + bs]).float().to(DEVICE)
        qb = torch.from_numpy(q[i:i + bs]).float().to(DEVICE)
        tb = torch.stack([encode_text(x) for x in t[i:i + bs]]).to(DEVICE)
        outs.append(torch.sigmoid(model(eb, qb, tb)).cpu().numpy())
    return np.concatenate(outs)


def run(model, tr_e, tr_q, tr_t, tr_y, va_e, va_q, va_t, va_y,
        epochs=5, bs=1024, lr=1e-3, noise=0.0, contrastive=False):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(tr_y)
    for ep in range(epochs):
        model.train()
        idx = np.arange(n)
        np.random.shuffle(idx)
        for s in range(0, n, bs):
            bi = idx[s:s + bs]
            eb = torch.from_numpy(tr_e[bi]).float().to(DEVICE)
            qb = torch.from_numpy(tr_q[bi]).float().to(DEVICE)
            tb = torch.stack([encode_text(x) for x in tr_t[bi]]).to(DEVICE)
            yb = torch.from_numpy(tr_y[bi]).float().to(DEVICE)
            if noise > 0:
                eb = eb + torch.randn_like(eb) * eb.std(1, keepdim=True) * noise
                qb = qb + torch.randn_like(qb) * qb.std(1, keepdim=True) * noise
            logit = model(eb, qb, tb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            if contrastive:
                # 附加: 拉近正对, 推开负对 (基于嵌入相似度)
                with torch.no_grad():
                    af = model.audio_fc(torch.cat([eb, qb, torch.abs(eb - qb), eb * qb], -1))
                pass
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        pr = predict(model, va_e, va_q, va_t)
        auc = roc_auc_score(va_y, pr)
        print(f"  ep{ep} loss={loss.item():.4f} 跨录音AUC={auc:.4f}", flush=True)
    return roc_auc_score(va_y, predict(model, va_e, va_q, va_t))


def main():
    enroll, query, labels, texts = load_data()
    n = len(labels)
    n_tr = int(n * 0.8)
    tr_e, tr_q, tr_y = enroll[:n_tr], query[:n_tr], labels[:n_tr]
    va_e, va_q, va_y = enroll[n_tr:], query[n_tr:], labels[n_tr:]
    tr_t = texts[:n_tr, 0]
    va_t = texts[n_tr:, 0]
    print(f"训练 {n_tr} 验证 {n - n_tr} pos={labels.mean():.3f}", flush=True)

    print("=== 基线 ===", flush=True)
    torch.manual_seed(SEED)
    auc0 = run(Matcher().to(DEVICE), tr_e, tr_q, tr_t, tr_y, va_e, va_q, va_t, va_y, epochs=5)
    print(f"基线 跨录音AUC={auc0:.4f}", flush=True)

    print("=== 噪声增强 (0.15) ===", flush=True)
    torch.manual_seed(SEED)
    auc1 = run(Matcher().to(DEVICE), tr_e, tr_q, tr_t, tr_y, va_e, va_q, va_t, va_y,
               epochs=5, noise=0.15)
    print(f"增强 跨录音AUC={auc1:.4f}", flush=True)


if __name__ == "__main__":
    main()
