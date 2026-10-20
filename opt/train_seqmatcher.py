"""SeqMatcher: frame-level audio matching + text, trained on val 10K for architecture validation."""
import sys, pickle, random, argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
FEATS = BASE / "output_whisper"
OUT = BASE / "opt" / "out_seq"
OUT.mkdir(parents=True, exist_ok=True)
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


class SeqMatcher(nn.Module):
    """Frame-level attention matching between enroll and query sequences."""
    def __init__(self, a_dim=384, proj=192, text_dim=128, hidden=512, n_frames=24):
        super().__init__()
        self.proj_e = nn.Sequential(nn.Linear(a_dim, proj), nn.LayerNorm(proj), nn.GELU())
        self.proj_q = nn.Sequential(nn.Linear(a_dim, proj), nn.LayerNorm(proj), nn.GELU())
        self.text_enc = TextEncoder()
        self.cls = nn.Sequential(
            nn.Linear(proj * 2 + 3 + 2 + text_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, ef, qf, eg, qg, text):
        """ef/qf: (B,24,384) frames; eg/qg: (B,384) global emb; text: (B,L)"""
        B = ef.shape[0]
        e = self.proj_e(ef)  # (B,24,proj)
        q = self.proj_q(qf)
        e = F.normalize(e, dim=-1)
        q = F.normalize(q, dim=-1)
        sim = e @ q.transpose(1, 2)  # (B,24,24) enroll x query
        # attended enroll per query frame
        att = torch.softmax(sim.transpose(1, 2), dim=-1)  # (B,24,24) query x enroll
        att_enroll = att @ e  # (B,24,proj)
        per_q = (att_enroll * q).sum(-1)  # (B,24) per-frame match
        topk = torch.topk(per_q, k=min(8, per_q.shape[1]), dim=1).values
        match_feat = torch.cat([
            per_q.mean(1, keepdim=True), per_q.max(1).values.unsqueeze(1), topk.mean(1, keepdim=True),
            att_enroll.mean(1), q.mean(1),
        ], dim=1)  # (B, proj*2+3)
        # global similarity features
        eg_n = F.normalize(eg, dim=-1); qg_n = F.normalize(qg, dim=-1)
        g_sim = (eg_n * qg_n).sum(-1, keepdim=True)
        g_diff = torch.abs(eg - qg).mean(-1, keepdim=True)
        tf = self.text_enc(text)
        x = torch.cat([match_feat, g_sim, g_diff, tf], dim=1)
        return self.cls(x).squeeze(-1)


def load_items():
    d = pickle.load(open(FEATS / "val_feats.pkl", "rb"))
    items = []
    for name in ["dev_seen", "dev_unseen"]:
        for pid, it in d[name].items():
            items.append({
                "pid": pid, "split": name,
                "ef": torch.tensor(it["enroll_frames"].astype(np.float32)),
                "qf": torch.tensor(it["query_frames"].astype(np.float32)),
                "eg": torch.tensor(it["enroll_emb"], dtype=torch.float32),
                "qg": torch.tensor(it["query_emb"], dtype=torch.float32),
                "text": encode_text(it["enroll_txt"]),
                "label": float(it["label"]),
            })
    return items


def make_batch(items, device, emb_noise=0.0, noise_scale=0.05):
    ef = torch.stack([it["ef"] for it in items]).to(device)
    qf = torch.stack([it["qf"] for it in items]).to(device)
    eg = torch.stack([it["eg"] for it in items]).to(device)
    qg = torch.stack([it["qg"] for it in items]).to(device)
    t = torch.stack([it["text"] for it in items]).to(device)
    y = torch.tensor([it["label"] for it in items], dtype=torch.float32).to(device)
    if emb_noise > 0:
        eg = eg + torch.randn_like(eg) * (eg.std(1, keepdim=True) + 1e-6) * noise_scale
        qg = qg + torch.randn_like(qg) * (qg.std(1, keepdim=True) + 1e-6) * noise_scale
        ef = ef + torch.randn_like(ef) * 0.02
        qf = qf + torch.randn_like(qf) * 0.02
    return ef, qf, eg, qg, t, y


@torch.no_grad()
def evaluate(model, items, device, bs=512):
    model.eval()
    preds, labels = [], []
    for i in range(0, len(items), bs):
        ef, qf, eg, qg, t, y = make_batch(items[i:i + bs], device)
        logit = model(ef, qf, eg, qg, t)
        preds.extend(torch.sigmoid(logit).cpu().flatten().tolist())
        labels.extend(y.cpu().tolist())
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, preds), preds


def train_val(tr, va, epochs=40, lr=1e-3, bs=256, tag="seq", patience=10, emb_noise=0.0):
    set_seed(SEED)
    model = SeqMatcher().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_ep, wait = 0.0, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        idx = list(range(len(tr))); random.shuffle(idx)
        for i in range(0, len(idx), bs):
            bi = [tr[j] for j in idx[i:i + bs]]
            ef, qf, eg, qg, t, y = make_batch(bi, DEVICE, emb_noise=emb_noise)
            logit = model(ef, qf, eg, qg, t)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        auc, _ = evaluate(model, va, DEVICE)
        print(f"  ep{ep:3d} loss={loss.item():.4f} val_AUC={auc:.4f}", flush=True)
        if auc > best:
            best, best_ep, wait = auc, ep, 0
            torch.save(model.state_dict(), OUT / f"best_{tag}.pt")
        else:
            wait += 1
            if wait >= patience:
                break
    return best, best_ep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="both", choices=["cv", "s2u", "both"])
    args = parser.parse_args()
    items = load_items()
    seen = [it for it in items if it["split"] == "dev_seen"]
    unseen = [it for it in items if it["split"] == "dev_unseen"]
    print(f"seen={len(seen)} unseen={len(unseen)}")

    results = {}
    if args.mode in ("s2u", "both"):
        print("\n=== seen -> unseen ===")
        auc, ep = train_val(seen, unseen, tag="s2u", epochs=40)
        results["seen2unseen"] = {"auc": auc, "ep": ep}
        print(f">>> dev_unseen AUC = {auc:.4f}")
    if args.mode in ("cv", "both"):
        print("\n=== 5-fold CV on all 10K ===")
        all_items = seen + unseen
        y_all = [it["label"] for it in all_items]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        folds = []
        for f, (tr_idx, va_idx) in enumerate(skf.split(all_items, y_all)):
            tr = [all_items[i] for i in tr_idx]
            va = [all_items[i] for i in va_idx]
            auc, _ = train_val(tr, va, tag=f"fold{f}", epochs=30, patience=6)
            folds.append(auc)
            print(f"  fold{f}: {auc:.4f}")
        results["cv5"] = {"mean": float(np.mean(folds)), "folds": folds}
        print(f">>> mean CV AUC = {np.mean(folds):.4f}")
    with open(OUT / "results_seq.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
