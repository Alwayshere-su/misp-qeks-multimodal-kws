"""
Phase 1: whisper 音频嵌入 + 文本模态 匹配模型
==============================================
特征: enroll_emb(384) + query_emb(384) + enroll_txt(字符编码)
模型: [enroll, query, diff, prod, text] -> MLP -> score

评估协议:
  1) dev_seen 训练 / dev_unseen 验证 (对照 V3: 0.6466)
  2) 全量 10K 5-fold 交叉验证
"""
import os, sys, pickle, random, argparse
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
OUT = BASE / "output_p1"
OUT.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# 字符表 (英文小写 + 标点)
CHARS = "abcdefghijklmnopqrstuvwxyz' -"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = pad
MAX_TXT = 24


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def encode_text(txt, max_len=MAX_TXT):
    """单词 -> (max_len,) token ids"""
    ids = [CHAR2IDX.get(c, 1) for c in txt.lower()]
    ids = ids[:max_len]
    ids = ids + [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class TextEncoder(nn.Module):
    """字符级 CNN -> 128 维"""
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
        """x: (B, L) -> (B, 128)"""
        e = self.emb(x).transpose(1, 2)       # (B, emb, L)
        h = self.conv(e)                       # (B, hidden, L)
        w = torch.softmax(h.mean(1, keepdim=True), -1)
        out = (w * h).sum(-1)                  # (B, hidden)
        return self.proj(out)


class MatcherV1(nn.Module):
    """[audio pair features + text] -> score"""
    def __init__(self, emb_dim=384, text_dim=128, hidden=512):
        super().__init__()
        self.text_enc = TextEncoder()
        # audio 特征: [enroll, query, diff, prod] = 4*384
        self.audio_fc = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
        )
        self.cls = nn.Sequential(
            nn.Linear(hidden + text_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, enroll_emb, query_emb, text_tokens):
        diff = torch.abs(enroll_emb - query_emb)
        prod = enroll_emb * query_emb
        af = self.audio_fc(torch.cat([enroll_emb, query_emb, diff, prod], -1))
        tf = self.text_enc(text_tokens)
        return self.cls(torch.cat([af, tf], -1))


def load_data():
    d = pickle.load(open(FEATS / "val_feats.pkl", 'rb'))
    items = []
    for name in ['dev_seen', 'dev_unseen']:
        for pid, it in d[name].items():
            items.append({
                'pid': pid,
                'split': name,
                'enroll_emb': torch.tensor(it['enroll_emb'], dtype=torch.float32),
                'query_emb': torch.tensor(it['query_emb'], dtype=torch.float32),
                'text': encode_text(it['enroll_txt']),
                'label': float(it['label']),
            })
    return items


def make_tensor_batch(items, device, emb_noise=0.0, noise_scale=0.05):
    e = torch.stack([it['enroll_emb'] for it in items]).to(device)
    q = torch.stack([it['query_emb'] for it in items]).to(device)
    t = torch.stack([it['text'] for it in items]).to(device)
    y = torch.tensor([it['label'] for it in items], dtype=torch.float32).to(device)
    if emb_noise > 0:
        scale = (e.std(dim=1, keepdim=True) + 1e-6) * noise_scale
        e = e + torch.randn_like(e) * scale
        q = q + torch.randn_like(q) * scale
    return e, q, t, y


def evaluate(model, items, device, bs=512):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            e, q, t, y = make_tensor_batch(items[i:i + bs], device)
            logit = model(e, q, t)
            preds.extend(torch.sigmoid(logit).cpu().flatten().tolist())
            labels.extend(y.cpu().tolist())
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return auc, preds


def train_val(train_items, val_items, epochs=60, lr=1e-3, bs=256, tag='p1', patience=12,
              emb_noise=0.0, noise_scale=0.05):
    set_seed(SEED)
    model = MatcherV1().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_auc, best_ep, wait = 0.0, 0, 0

    for ep in range(1, epochs + 1):
        model.train()
        idx = list(range(len(train_items)))
        random.shuffle(idx)
        for i in tqdm(range(0, len(idx), bs), desc=f"ep{ep}", leave=False):
            bi = [train_items[j] for j in idx[i:i + bs]]
            e, q, t, y = make_tensor_batch(bi, DEVICE, emb_noise=emb_noise, noise_scale=noise_scale)
            logit = model(e, q, t).squeeze(-1)
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
        model = MatcherV1().to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
        bs = 256
        for ep in range(1, 41):
            model.train()
            idx = list(range(len(items)))
            random.shuffle(idx)
            for i in range(0, len(idx), bs):
                bi = [items[j] for j in idx[i:i + bs]]
                e, q, t, y = make_tensor_batch(bi, DEVICE)
                logit = model(e, q, t).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logit, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sched.step()
            if ep % 5 == 0 or ep == 40:
                print(f"  ep{ep:3d} loss={loss.item():.4f}")
        torch.save(model.state_dict(), OUT / "best_full.pt")
        print("  saved output_p1/best_full.pt")
        return
    if args.mode in ('seen2unseen', 'both'):
        print("\n=== [1] dev_seen 训练 -> dev_unseen 验证 (对照 V3 0.6466) ===")
        auc, ep = train_val(seen, unseen, tag='s2u')
        results['seen2unseen'] = {'auc': auc, 'ep': ep}
        print(f"  >>> dev_unseen AUC = {auc:.4f} (V3=0.6466)")

    if args.mode in ('cv', 'both'):
        print("\n=== [2] 全量 10K 5-fold 交叉验证 ===")
        all_items = seen + unseen
        rng = random.Random(SEED)
        # 按 split 分层: 每折保留相同比例
        from sklearn.model_selection import StratifiedKFold
        y_all = [it['label'] for it in all_items]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_aucs = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(all_items, y_all)):
            tr = [all_items[i] for i in tr_idx]
            va = [all_items[i] for i in va_idx]
            auc, ep = train_val(tr, va, epochs=40, patience=8, tag=f'fold{fold}')
            fold_aucs.append(auc)
            print(f"  fold{fold}: AUC={auc:.4f}")
        mean_auc = float(np.mean(fold_aucs))
        results['cv5'] = {'mean_auc': mean_auc, 'folds': fold_aucs}
        print(f"  >>> 5-fold mean AUC = {mean_auc:.4f}")

    with open(OUT / "results_p1.json", 'w') as f:
        import json
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    run()
