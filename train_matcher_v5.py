"""
MatcherV5 — 用完整 train 500K 对训练 (audio + text), val 验证
===============================================================
输入特征: output_whisper/train_feats.npz (enroll_emb, query_emb, labels, texts)
验证:     output_whisper/val_feats.pkl (dev_seen / dev_unseen 分开报告)
模型:     MatcherV1 结构放大版 [enroll, query, diff, prod] + text -> MLP -> score
用法:     python train_matcher_v5.py [--epochs 15] [--bs 1024] [--tag v5]
"""
import os, sys, pickle, random, argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
FEATS = BASE / "output_whisper"
OUT = BASE / "output_v5"
OUT.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

CHARS = "abcdefghijklmnopqrstuvwxyz' -"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
MAX_TXT = 24


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def encode_text(txt, max_len=MAX_TXT):
    ids = [CHAR2IDX.get(c, 1) for c in txt.lower()][:max_len]
    ids = ids + [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class TextEncoder(nn.Module):
    def __init__(self, emb_dim=64, hidden=128, num_chars=len(CHARS) + 1):
        super().__init__()
        self.emb = nn.Embedding(num_chars, emb_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = self.conv(e)
        w = torch.softmax(h.mean(1, keepdim=True), -1)
        return self.proj((w * h).sum(-1))


class MatcherV5(nn.Module):
    def __init__(self, a_dim=384, text_dim=128, hidden=512):
        super().__init__()
        self.text_enc = TextEncoder()
        self.audio_fc = nn.Sequential(
            nn.Linear(a_dim * 4, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
        )
        self.cls = nn.Sequential(
            nn.Linear(hidden + text_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, ea, qa, text):
        diff = torch.abs(ea - qa)
        prod = ea * qa
        af = self.audio_fc(torch.cat([ea, qa, diff, prod], -1))
        tf = self.text_enc(text)
        return self.cls(torch.cat([af, tf], -1))


def load_train(npz):
    d = np.load(npz, allow_pickle=True)
    enroll = torch.tensor(d['enroll_emb'], dtype=torch.float32)
    query = torch.tensor(d['query_emb'], dtype=torch.float32)
    labels = torch.tensor(d['labels'], dtype=torch.float32)
    texts = [encode_text(t) for t in d['texts'][:, 0]]  # 只用 enroll_txt
    return enroll, query, labels, texts


def load_val():
    d = pickle.load(open(FEATS / "val_feats.pkl", 'rb'))
    items = []
    for name in ['dev_seen', 'dev_unseen']:
        for pid, it in d[name].items():
            items.append({
                'pid': pid, 'split': name,
                'enroll_emb': torch.tensor(it['enroll_emb'], dtype=torch.float32),
                'query_emb': torch.tensor(it['query_emb'], dtype=torch.float32),
                'text': encode_text(it['enroll_txt']),
                'label': float(it['label']),
            })
    return items


@torch.no_grad()
def evaluate(model, items, bs=1024):
    model.eval()
    preds, labels = [], []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        ea = torch.stack([it['enroll_emb'] for it in chunk]).to(DEVICE)
        qa = torch.stack([it['query_emb'] for it in chunk]).to(DEVICE)
        t = torch.stack([it['text'] for it in chunk]).to(DEVICE)
        logit = model(ea, qa, t)
        preds.extend(torch.sigmoid(logit).cpu().flatten().tolist())
        labels.extend([it['label'] for it in chunk])
    return preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--emb_noise', type=float, default=0.05)
    parser.add_argument('--tag', default='v5')
    parser.add_argument('--npz', default=str(FEATS / "train_feats.npz"))
    args = parser.parse_args()

    set_seed(SEED)
    print(f"Device: {DEVICE} | epochs={args.epochs} bs={args.bs} lr={args.lr}")

    print("[1] 加载 train 特征...")
    train_e, train_q, train_y, train_t = load_train(args.npz)
    print(f"  train: {train_e.shape[0]} pairs, pos rate={train_y.mean():.3f}")

    print("[2] 加载 val 特征...")
    val_items = load_val()
    seen = [it for it in val_items if it['split'] == 'dev_seen']
    unseen = [it for it in val_items if it['split'] == 'dev_unseen']
    print(f"  val: {len(seen)} seen / {len(unseen)} unseen")

    model = MatcherV5().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n = train_e.shape[0]
    best_score, best_ep = 0.0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        idx = list(range(n)); random.shuffle(idx)
        tot_loss = 0.0; nb = 0
        for i in tqdm(range(0, n, args.bs), desc=f"ep{ep}", leave=False):
            bi = idx[i:i + args.bs]
            ea = train_e[bi].to(DEVICE)
            qa = train_q[bi].to(DEVICE)
            t = torch.stack([train_t[j] for j in bi]).to(DEVICE)
            y = train_y[bi].to(DEVICE)
            if args.emb_noise > 0:
                scale = (ea.std(1, keepdim=True) + 1e-6) * args.emb_noise
                ea = ea + torch.randn_like(ea) * scale
                qa = qa + torch.randn_like(qa) * scale
            logit = model(ea, qa, t).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += loss.item(); nb += 1
        sched.step()

        preds, labels = evaluate(model, val_items)
        ps, ls = [], []
        for p, it in zip(preds, val_items):
            ps.append(p); ls.append(it['label'])
        auc_all = roc_auc_score(ls, ps)
        auc_seen = roc_auc_score([it['label'] for it in seen],
                                 [p for p, it in zip(preds, val_items) if it['split'] == 'dev_seen'])
        auc_unseen = roc_auc_score([it['label'] for it in unseen],
                                   [p for p, it in zip(preds, val_items) if it['split'] == 'dev_unseen'])
        score = 0.5 * auc_seen + 0.5 * auc_unseen
        print(f"  ep{ep:3d} loss={tot_loss/nb:.4f} AUC_all={auc_all:.4f} "
              f"seen={auc_seen:.4f} unseen={auc_unseen:.4f} score={score:.4f}")
        if score > best_score:
            best_score, best_ep = score, ep
            torch.save(model.state_dict(), OUT / f"best_{args.tag}.pt")
            print(f"    * saved best_{args.tag}.pt (score={score:.4f})")

    print(f"\n[DONE] best score={best_score:.4f} @ ep{best_ep}")
    json.dump({'best_score': best_score, 'best_ep': best_ep},
              open(OUT / f"results_{args.tag}.json", 'w'), indent=2)


if __name__ == '__main__':
    main()
