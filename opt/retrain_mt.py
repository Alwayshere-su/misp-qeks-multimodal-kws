"""重新训练: MatcherV4 + 多模板特征 (全局 + 帧级), val 10K 训练, test 提交
多模板特征 = rank 归一化 (消除 val 5模板 vs test 23模板 的分布差异)
"""
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BASE = Path(r"D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
sys.path.insert(0, str(BASE))
from train_whisper_p4 import MatcherV4, encode_text

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
OUT = BASE / "opt" / "out_retrain"
OUT.mkdir(parents=True, exist_ok=True)


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class MatcherMT(nn.Module):
    """MatcherV4 基础上加入多模板特征 (2维 rank)"""

    def __init__(self, mt_dim=2):
        super().__init__()
        self.base = MatcherV4()
        # MatcherV4.cls 输入 = hidden*2 + text_dim = 512*2+128 = 1152
        self.mt_fc = nn.Sequential(nn.Linear(mt_dim, 32), nn.GELU())
        self.cls = nn.Sequential(
            nn.Linear(1152 + 32, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, ea, qa, ev, qv, text, mt):
        # 复用 MatcherV4 的 audio/video/text 编码
        a_diff = torch.abs(ea - qa)
        a_prod = ea * qa
        af = self.base.audio_fc(torch.cat([ea, qa, a_diff, a_prod], -1))
        v_diff = torch.abs(ev - qv)
        v_prod = ev * qv
        vf = self.base.video_fc(torch.cat([ev, qv, v_diff, v_prod], -1))
        tf = self.base.text_enc(text)
        base_feat = torch.cat([af, vf, tf], -1)  # 640
        mf = self.mt_fc(mt)
        return self.cls(torch.cat([base_feat, mf], -1)).squeeze(-1)


def compute_mt(data, use_frame=False):
    """每词模板的全局/帧级多模板分数 (rank 归一化)"""
    import re
    def norm(s):
        return re.sub(r"[^a-z0-9']", '', (s or '').lower())
    from collections import defaultdict
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))
    raw = {}
    for w, items in word_items.items():
        if use_frame:
            T = np.concatenate([it['enroll_frames'].astype(np.float32) for _, it in items], axis=0)
            Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
            for pid, it in items:
                qf = it['query_frames'].astype(np.float32)
                qn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-9)
                raw[pid] = float((qn @ Tn.T).max(axis=1).mean())
        else:
            T = np.stack([it['enroll_emb'] for _, it in items])
            Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
            for pid, it in items:
                q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
                raw[pid] = float((q @ Tn.T).max())
    # rank 归一化
    from scipy.stats import rankdata
    pids = sorted(raw.keys())
    r = rankdata([raw[p] for p in pids]) / len(pids)
    return dict(zip(pids, r))


def load_train():
    vd = pickle.load(open(BASE / "output_whisper/val_feats.pkl", "rb"))
    lip = pickle.load(open(BASE / "output_lip/val_lip.pkl", "rb"))
    mt_g = {}
    mt_f = {}
    for split in ["dev_seen", "dev_unseen"]:
        mt_g.update(compute_mt(vd[split], use_frame=False))
        mt_f.update(compute_mt(vd[split], use_frame=True))
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
                "mt": torch.tensor([mt_g.get(pid, 0.5), mt_f.get(pid, 0.5)], dtype=torch.float32),
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


@torch.no_grad()
def predict(model, items, device, bs=512):
    model.eval()
    outs = []
    for i in range(0, len(items), bs):
        ea, qa, ev, qv, t, mt, _ = make_batch(items[i:i + bs], device)
        logit = model(ea, qa, ev, qv, t, mt)
        outs.extend(torch.sigmoid(logit).cpu().flatten().tolist())
    return dict(zip([it["pid"] for it in items], outs))


def main():
    set_seed(SEED)
    items = load_train()
    print(f"val items: {len(items)}", flush=True)
    model = MatcherMT().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    n = len(items)
    for ep in range(1, 41):
        model.train()
        idx = list(range(n))
        random.shuffle(idx)
        for i in range(0, n, 256):
            bi = [items[j] for j in idx[i:i + 256]]
            ea, qa, ev, qv, t, mt, y = make_batch(bi, DEVICE, noise=0.08)
            logit = model(ea, qa, ev, qv, t, mt)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        if ep % 5 == 0 or ep == 40:
            print(f"  ep{ep} loss={loss.item():.4f}", flush=True)
    torch.save(model.state_dict(), OUT / "best_mt.pt")
    print("saved best_mt.pt", flush=True)

    # test 预测
    tin = pickle.load(open(BASE / "output_whisper/test_inset.pkl", "rb"))
    tout = pickle.load(open(BASE / "output_whisper/test_outset.pkl", "rb"))
    lip_in = pickle.load(open(BASE / "output_lip/test_lip_inset.pkl", "rb"))
    lip_out = pickle.load(open(BASE / "output_lip/test_lip_outset.pkl", "rb"))
    mt_g_i = compute_mt(tin, False)
    mt_f_i = compute_mt(tin, True)
    mt_g_o = compute_mt(tout, False)
    mt_f_o = compute_mt(tout, True)

    def build_test(d, lip, mg, mf):
        items = []
        for pid, it in d.items():
            v = lip.get(pid, {})
            items.append({
                "pid": pid,
                "enroll_emb": torch.tensor(it["enroll_emb"], dtype=torch.float32),
                "query_emb": torch.tensor(it["query_emb"], dtype=torch.float32),
                "enroll_video": torch.tensor(v.get("enroll_video", np.zeros(256, np.float32)), dtype=torch.float32),
                "query_video": torch.tensor(v.get("query_video", np.zeros(256, np.float32)), dtype=torch.float32),
                "text": encode_text(it["enroll_txt"]),
                "mt": torch.tensor([mg.get(pid, 0.5), mf.get(pid, 0.5)], dtype=torch.float32),
                "label": 0.0,
            })
        return items

    print("test 预测...", flush=True)
    pi = predict(model, build_test(tin, lip_in, mt_g_i, mt_f_i), DEVICE)
    po = predict(model, build_test(tout, lip_out, mt_g_o, mt_f_o), DEVICE)
    with open(BASE / "submission/submit_retrain_mt.csv", "w", encoding="utf-8", newline="") as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},inset,{pi.get(pid, 0.5):.6f}\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},outset,{po.get(pid, 0.5):.6f}\n")
    print("saved submission/submit_retrain_mt.csv", flush=True)


if __name__ == "__main__":
    main()
