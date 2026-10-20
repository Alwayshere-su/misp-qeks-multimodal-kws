"""base + tiny 多模板融合验证 (val 抽样)"""
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def main():
    import whisper
    model = whisper.load_model('base')
    model.eval()
    vd = pickle.load(open(BASE / "output_whisper/val_feats.pkl", "rb"))
    import random
    random.seed(0)
    items = []
    for split in ['dev_seen', 'dev_unseen']:
        pids = random.sample(sorted(vd[split].keys()), 800)
        for pid in pids:
            items.append((split, pid, vd[split][pid]))

    # base 嵌入
    e_emb, q_emb = {}, {}
    from base_small_mt import embed_whisper
    for i in range(0, len(items), 32):
        chunk = items[i:i + 32]
        e_b, q_b, e_vl, q_vl = [], [], [], []
        for split, pid, it in chunk:
            for kind, lst, vl in [('enroll', e_b, e_vl), ('query', q_b, q_vl)]:
                wav = str(BASE / f'val/{split}/wav/{pid}_{kind}.wav')
                d, sr = sf.read(wav, dtype='float32')
                a = torch.from_numpy(d).float()
                lst.append(a)
                vl.append(a.shape[0])
        e_embs = embed_whisper(model, e_b, e_vl)
        q_embs = embed_whisper(model, q_b, q_vl)
        for j, (split, pid, _) in enumerate(chunk):
            e_emb[pid] = e_embs[j]
            q_emb[pid] = q_embs[j]
        if (i // 32) % 10 == 0:
            print(f'  {i + len(chunk)}/{len(items)}', flush=True)

    from collections import defaultdict
    for split in ['dev_seen', 'dev_unseen']:
        sub = [(s, p, it) for s, p, it in items if s == split]
        wi = defaultdict(list)
        for s, p, it in sub:
            wi[norm(it['enroll_txt'])].append((s, p, it))
        s_tiny, s_base, ys = [], [], []
        for w, subs in wi.items():
            Tb = np.stack([e_emb[p] for _, p, _ in subs])
            Tbn = Tb / (np.linalg.norm(Tb, axis=1, keepdims=True) + 1e-9)
            Tt = np.stack([it['enroll_emb'] for _, _, it in subs])
            Ttn = Tt / (np.linalg.norm(Tt, axis=1, keepdims=True) + 1e-9)
            for s, p, it in subs:
                qb = q_emb[p] / (np.linalg.norm(q_emb[p]) + 1e-9)
                qt = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
                s_base.append(float((qb @ Tbn.T).max()))
                s_tiny.append(float((qt @ Ttn.T).max()))
                ys.append(it['label'])
        a, b = np.array(s_tiny), np.array(s_base)
        ar, br = rankdata(a) / len(a), rankdata(b) / len(b)
        print(f'=== {split} ===', flush=True)
        print(f'  tiny: {roc_auc_score(ys, a):.4f}  base: {roc_auc_score(ys, b):.4f}', flush=True)
        best = 0
        for w in np.arange(0, 1.01, 0.1):
            auc = roc_auc_score(ys, w * ar + (1 - w) * br)
            if auc > best:
                best, bw = auc, w
        print(f'  融合 best w_tiny={bw:.1f} AUC={best:.4f}', flush=True)


if __name__ == '__main__':
    main()
