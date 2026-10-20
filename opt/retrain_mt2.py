"""改进重训练: 丰富多模板特征 (g_max/g_top2/g_norm/f_max/f_top2) + early stop + best_full 初始化
训练小融合器 + 完整 MatcherMT 两个版本
"""
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score

BASE = Path(r"D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
sys.path.insert(0, str(BASE))
from train_whisper_p4 import MatcherV4, encode_text

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
OUT = BASE / "opt" / "out_retrain"
OUT.mkdir(parents=True, exist_ok=True)


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def compute_mt_feats(data):
    """丰富多模板特征: dict[pid] = np.array([g_max, g_top2, g_norm, f_max, f_top2])"""
    from collections import defaultdict
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))
    all_emb = np.stack([it['enroll_emb'] for items in word_items.values() for _, it in items])
    all_mean_n = all_emb.mean(0)
    all_mean_n = all_mean_n / (np.linalg.norm(all_mean_n) + 1e-9)
    feats = {}
    for w, items in word_items.items():
        Tg = np.stack([it['enroll_emb'] for _, it in items])
        Tgn = Tg / (np.linalg.norm(Tg, axis=1, keepdims=True) + 1e-9)
        Tf = np.concatenate([it['enroll_frames'].astype(np.float32) for _, it in items], axis=0)
        Tfn = Tf / (np.linalg.norm(Tf, axis=1, keepdims=True) + 1e-9)
        for pid, it in items:
            q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
            sims = q @ Tgn.T
            g_max = float(sims.max())
            g_top2 = float(np.sort(sims)[-2:].mean())
            g_norm = float(g_max - (q @ all_mean_n))
            qf = it['query_frames'].astype(np.float32)
            qfn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-9)
            fsim = qfn @ Tfn.T
            f_max = float(fsim.max(axis=1).mean())
            f_top2 = float(np.sort(fsim.max(axis=1))[-2:].mean())
            feats[pid] = np.array([g_max, g_top2, g_norm, f_max, f_top2], dtype=np.float32)
    return feats


class MatcherMT2(nn.Module):
    def __init__(self, mt_dim=5):
        super().__init__()
        self.base = MatcherV4()
        self.mt_fc = nn.Sequential(nn.Linear(mt_dim, 64), nn.GELU())
        self.cls = nn.Sequential(
            nn.Linear(1152 + 64, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, ea, qa, ev, qv, text, mt):
        a_diff = torch.abs(ea - qa)
        af = self.base.audio_fc(torch.cat([ea, qa, a_diff, ea * qa], -1))
        vf = self.base.video_fc(torch.cat([ev, qv, torch.abs(ev - qv), ev * qv], -1))
        tf = self.base.text_enc(text)
        base_feat = torch.cat([af, vf, tf], -1)
        mf = self.mt_fc(mt)
        return self.cls(torch.cat([base_feat, mf], -1)).squeeze(-1)


def load_items(vd, lip, feats):
    items = []
    for name in ["dev_seen", "dev_unseen"]:
        for pid, it in vd[name].items():
            v = lip[name].get(pid, {})
            items.append({
                "pid": pid, "split": name,
                "enroll_emb": torch.tensor(it["enroll_emb"], dtype=torch.float32),
                "query_emb": torch.tensor(it["query_emb"], dtype=torch.float32),
                "enroll_video": torch.tensor(v.get("enroll_video", np.zeros(256, np.float32)), dtype=torch.float32),
                "query_video": torch.tensor(v.get("query_video", np.zeros(256, np.float32)), dtype=torch.float32),
                "text": encode_text(it["enroll_txt"]),
                "mt": torch.tensor(feats[pid], dtype=torch.float32),
                "label": float(it["label"]),
            })
    return items


def make_batch(items, device, bs=256, noise=0.0):
    ea = torch.stack([it["enroll_emb"] for it in items]).to(device)
    qa = torch.stack([it["query_emb"] for it in items]).to(device)
    ev = torch.stack([it["enroll_video"] for it in items]).to(device)
    qv = torch.stack([it["query_video"] for it in items]).to(device)
    t = torch.stack([it["text"] for it in items]).to(device)
    mt = torch.stack([it["mt"] for it in items]).to(device)
    y = torch.tensor([it["label"] for it in items], dtype=torch.float32).to(device)
    if noise > 0:
        ea = ea + torch.randn_like(ea) * ea.std(1, keepdim=True) * noise
        qa = qa + torch.randn_like(qa) * qa.std(1, keepdim=True) * noise
    return ea, qa, ev, qv, t, mt, y


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    vd = pickle.load(open(BASE / "output_whisper/val_feats.pkl", "rb"))
    lip = pickle.load(open(BASE / "output_lip/val_lip.pkl", "rb"))
    feats = {}
    for split in ["dev_seen", "dev_unseen"]:
        feats.update(compute_mt_feats(vd[split]))
    items = load_items(vd, lip, feats)
    print(f"items: {len(items)}, 特征维度: {len(next(iter(feats.values())))}", flush=True)

    # 训练/验证 split (10% 验证)
    idx = list(range(len(items)))
    random.shuffle(idx)
    n_va = len(items) // 10
    va = [items[i] for i in idx[:n_va]]
    tr = [items[i] for i in idx[n_va:]]

    model = MatcherMT2().to(DEVICE)
    # best_full 初始化 base
    try:
        sd = torch.load(BASE / "output_p4/best_full.pt", map_location="cpu", weights_only=True)
        base_sd = {k[len('base.'):] if k.startswith('base.') else k: v
                   for k, v in sd.items()}
        missing, unexpected = model.base.load_state_dict(base_sd, strict=False)
        print(f"best_full 初始化: missing={len(missing)}", flush=True)
    except Exception as e:
        print(f"初始化失败: {e}", flush=True)

    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    best_auc, best_ep, wait = 0.0, 0, 0
    for ep in range(1, 31):
        model.train()
        ridx = list(range(len(tr)))
        random.shuffle(ridx)
        for i in range(0, len(ridx), 256):
            bi = [tr[j] for j in ridx[i:i + 256]]
            ea, qa, ev, qv, t, mt, y = make_batch(bi, DEVICE, noise=0.05)
            logit = model(ea, qa, ev, qv, t, mt)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for i in range(0, len(va), 256):
                ea, qa, ev, qv, t, mt, y = make_batch(va[i:i + 256], DEVICE)
                logit = model(ea, qa, ev, qv, t, mt)
                preds.extend(torch.sigmoid(logit).cpu().flatten().tolist())
                ys.extend(y.cpu().tolist())
            auc = roc_auc_score(ys, preds)
        print(f"  ep{ep} loss={loss.item():.4f} val_auc={auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, best_ep, wait = auc, ep, 0
            torch.save(model.state_dict(), OUT / "best_mt2.pt")
        else:
            wait += 1
            if wait >= 6:
                print(f"early stop @ {ep}, best={best_auc:.4f} @ ep{best_ep}", flush=True)
                break
    print(f"best val_auc={best_auc:.4f}", flush=True)

    # test 预测
    model.load_state_dict(torch.load(OUT / "best_mt2.pt", map_location="cpu", weights_only=True))
    model.eval()
    tin = pickle.load(open(BASE / "output_whisper/test_inset.pkl", "rb"))
    tout = pickle.load(open(BASE / "output_whisper/test_outset.pkl", "rb"))
    lip_in = pickle.load(open(BASE / "output_lip/test_lip_inset.pkl", "rb"))
    lip_out = pickle.load(open(BASE / "output_lip/test_lip_outset.pkl", "rb"))
    ft_i = compute_mt_feats(tin)
    ft_o = compute_mt_feats(tout)

    def build_test(d, lipd, fts):
        return [{
            "pid": pid,
            "enroll_emb": torch.tensor(it["enroll_emb"], dtype=torch.float32),
            "query_emb": torch.tensor(it["query_emb"], dtype=torch.float32),
            "enroll_video": torch.tensor(lipd.get(pid, {}).get("enroll_video", np.zeros(256, np.float32)), dtype=torch.float32),
            "query_video": torch.tensor(lipd.get(pid, {}).get("query_video", np.zeros(256, np.float32)), dtype=torch.float32),
            "text": encode_text(it["enroll_txt"]),
            "mt": torch.tensor(fts[pid], dtype=torch.float32),
            "label": 0.0,
        } for pid, it in d.items()]

    @torch.no_grad()
    def pred(items):
        outs = {}
        for i in range(0, len(items), 512):
            ea, qa, ev, qv, t, mt, _ = make_batch(items[i:i + 512], DEVICE)
            logit = model(ea, qa, ev, qv, t, mt)
            for it, p in zip(items[i:i + 512], torch.sigmoid(logit).cpu().flatten().tolist()):
                outs[it["pid"]] = p
        return outs

    print("test 预测...", flush=True)
    pi = pred(build_test(tin, lip_in, ft_i))
    po = pred(build_test(tout, lip_out, ft_o))
    with open(BASE / "submission/submit_retrain_mt2.csv", "w", encoding="utf-8", newline="") as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},inset,{pi.get(pid, 0.5):.6f}\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},outset,{po.get(pid, 0.5):.6f}\n")
    print("saved submission/submit_retrain_mt2.csv", flush=True)


if __name__ == "__main__":
    main()
