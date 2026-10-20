"""
Phase 2: MatcherV2 — whisper 帧级 cross-attention 匹配
======================================================
特征: enroll_emb(384) + enroll_frames(24,384) + query_emb(384) + query_frames(24,384) + enroll_txt
模型:
  1) 双向 cross-attention: enroll 全局 attend query 帧; query 全局 attend enroll 帧
  2) 帧级相似度特征: 行/列 max-mean, 全局 max/mean
  3) 全局对 + 文本 -> MLP -> score
"""
import os, sys, pickle, random, argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
FEATS = BASE / "output_whisper"
OUT = BASE / "output_p2"
OUT.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

CHARS = "abcdefghijklmnopqrstuvwxyz' -"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
MAX_TXT = 24


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


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


class MatcherV2(nn.Module):
    def __init__(self, emb_dim=384, n_frames=24, text_dim=128, hidden=512):
        super().__init__()
        self.text_enc = TextEncoder()
        # 双向 cross-attention
        self.attn_eq = nn.MultiheadAttention(emb_dim, 6, batch_first=True, dropout=0.1)
        self.attn_qe = nn.MultiheadAttention(emb_dim, 6, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(emb_dim)
        # 全局特征 MLP: [e, q, e_attn, q_attn, diff, prod] = 6*384
        self.global_fc = nn.Sequential(
            nn.Linear(emb_dim * 6, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
        )
        # 分类: 全局 + 帧相似度特征(7) + 文本
        self.cls = nn.Sequential(
            nn.Linear(hidden + 7 + text_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, e_emb, e_frames, q_emb, q_frames, text):
        """
        e_emb/q_emb: (B,384); e_frames/q_frames: (B,24,384); text: (B,L)
        """
        B = e_emb.shape[0]
        # 双向 cross-attention
        e_attn, _ = self.attn_eq(e_emb.unsqueeze(1), q_frames, q_frames)   # (B,1,384)
        q_attn, _ = self.attn_qe(q_emb.unsqueeze(1), e_frames, e_frames)
        e_attn = self.norm(e_attn.squeeze(1))
        q_attn = self.norm(q_attn.squeeze(1))

        # 帧级相似度特征
        ef = F.normalize(e_frames.float(), dim=-1)
        qf = F.normalize(q_frames.float(), dim=-1)
        sim = ef @ qf.transpose(1, 2)                      # (B,24,24)
        fr_feat = torch.stack([
            sim.max(dim=2).values.mean(dim=1),   # 行 max 均值
            sim.max(dim=1).values.mean(dim=1),   # 列 max 均值
            sim.mean(dim=(1, 2)),
            sim.max(dim=1).values.max(dim=1).values,
            (sim > 0.8).float().mean(dim=(1, 2)),
            (sim > 0.9).float().mean(dim=(1, 2)),
            (sim > 0.95).float().mean(dim=(1, 2)),
        ], dim=1)                                  # (B,7)

        diff = torch.abs(e_emb - q_attn)
        prod = e_emb * q_attn
        gf = self.global_fc(torch.cat([e_emb, q_emb, e_attn, q_attn, diff, prod], -1))
        tf = self.text_enc(text)
        return self.cls(torch.cat([gf, fr_feat, tf], -1))


def load_data():
    d = pickle.load(open(FEATS / "val_feats.pkl", 'rb'))
    items = []
    for name in ['dev_seen', 'dev_unseen']:
        for pid, it in d[name].items():
            items.append({
                'pid': pid, 'split': name,
                'enroll_emb': torch.tensor(it['enroll_emb'], dtype=torch.float32),
                'enroll_frames': torch.tensor(it['enroll_frames'], dtype=torch.float32),
                'query_emb': torch.tensor(it['query_emb'], dtype=torch.float32),
                'query_frames': torch.tensor(it['query_frames'], dtype=torch.float32),
                'text': encode_text(it['enroll_txt']),
                'label': float(it['label']),
            })
    return items


def make_batch(items, device):
    e = torch.stack([it['enroll_emb'] for it in items]).to(device)
    ef = torch.stack([it['enroll_frames'] for it in items]).to(device)
    q = torch.stack([it['query_emb'] for it in items]).to(device)
    qf = torch.stack([it['query_frames'] for it in items]).to(device)
    t = torch.stack([it['text'] for it in items]).to(device)
    y = torch.tensor([it['label'] for it in items], dtype=torch.float32).to(device)
    return e, ef, q, qf, t, y


def evaluate(model, items, device, bs=512):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            e, ef, q, qf, t, y = make_batch(items[i:i + bs], device)
            logit = model(e, ef, q, qf, t)
            preds.extend(torch.sigmoid(logit).cpu().flatten().tolist())
            labels.extend(y.cpu().tolist())
    return roc_auc_score(labels, preds), preds


def train_val(train_items, val_items, epochs=60, lr=1e-3, bs=256, tag='p2', patience=12):
    set_seed(SEED)
    model = MatcherV2().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_auc, best_ep, wait = 0.0, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        idx = list(range(len(train_items)))
        random.shuffle(idx)
        for i in range(0, len(idx), bs):
            bi = [train_items[j] for j in idx[i:i + bs]]
            e, ef, q, qf, t, y = make_batch(bi, DEVICE)
            logit = model(e, ef, q, qf, t).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        v_auc, _ = evaluate(model, val_items, DEVICE)
        print(f"  ep{ep:3d} loss={loss.item():.4f} val_AUC={v_auc:.4f}")
        if v_auc > best_auc:
            best_auc, best_ep, wait = v_auc, ep, 0
            torch.save(model.state_dict(), OUT / f"best_{tag}.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop @ {ep}")
                break
    return best_auc, best_ep


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='both', choices=['seen2unseen', 'cv', 'both', 'full'])
    args = parser.parse_args()

    items = load_data()
    seen = [it for it in items if it['split'] == 'dev_seen']
    unseen = [it for it in items if it['split'] == 'dev_unseen']
    print(f"dev_seen: {len(seen)} | dev_unseen: {len(unseen)} | pos rate: "
          f"{sum(it['label'] for it in items)/len(items):.2f}")

    results = {}
    if args.mode == 'full':
        print("\n=== 全量 10K 训练 (无验证, 固定 40 epochs) ===")
        set_seed(SEED)
        model = MatcherV2().to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
        bs = 256
        for ep in range(1, 41):
            model.train()
            idx = list(range(len(items))); random.shuffle(idx)
            for i in range(0, len(idx), bs):
                bi = [items[j] for j in idx[i:i + bs]]
                e, ef, q, qf, t, y = make_batch(bi, DEVICE)
                logit = model(e, ef, q, qf, t).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logit, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sched.step()
            if ep % 5 == 0 or ep == 40:
                print(f"  ep{ep:3d} loss={loss.item():.4f}")
        torch.save(model.state_dict(), OUT / "best_full.pt")
        print("  saved output_p2/best_full.pt")
        return

    if args.mode in ('seen2unseen', 'both'):
        print("\n=== [1] dev_seen 训练 -> dev_unseen 验证 ===")
        auc, ep = train_val(seen, unseen, tag='s2u')
        results['seen2unseen'] = {'auc': auc, 'ep': ep}
        print(f"  >>> dev_unseen AUC = {auc:.4f} (V3=0.6466, P1=0.9601)")

    if args.mode in ('cv', 'both'):
        print("\n=== [2] 全量 10K 5-fold 交叉验证 ===")
        all_items = seen + unseen
        y_all = [it['label'] for it in all_items]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_aucs = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(all_items, y_all)):
            tr = [all_items[i] for i in tr_idx]
            va = [all_items[i] for i in va_idx]
            auc, _ = train_val(tr, va, epochs=40, patience=8, tag=f'fold{fold}')
            fold_aucs.append(auc)
            print(f"  fold{fold}: AUC={auc:.4f}")
        results['cv5'] = {'mean_auc': float(np.mean(fold_aucs)), 'folds': fold_aucs}
        print(f"  >>> 5-fold mean AUC = {np.mean(fold_aucs):.4f}")

    with open(OUT / "results_p2.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    run()
