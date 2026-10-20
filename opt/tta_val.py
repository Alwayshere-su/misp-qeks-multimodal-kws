"""TTA 验证: query 音频加噪多次嵌入平均 -> 多模板 AUC (val 抽样)"""
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import roc_auc_score

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')
sys.path.insert(0, str(BASE))
from whisper_encoder import WhisperEncoder

CKPT = BASE / "辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt"


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def main():
    enc = WhisperEncoder(CKPT).to("cuda")
    enc.eval()
    vd = pickle.load(open(BASE / "output_whisper/val_feats.pkl", "rb"))
    import random
    random.seed(1)
    # 抽样: seen 1000 + unseen 1000
    items = []
    for split in ['dev_seen', 'dev_unseen']:
        pids = random.sample(sorted(vd[split].keys()), 1000)
        for pid in pids:
            items.append((split, pid, vd[split][pid]))
    print(f'抽样 {len(items)}', flush=True)

    # 原始嵌入 (来自 val_feats, 与 TTA 对比)
    from collections import defaultdict
    word_items = defaultdict(list)
    for split, pid, it in items:
        word_items[norm(it['enroll_txt'])].append((split, pid, it))

    def mt_auc(get_emb):
        scores = {}
        y = []
        for w, subs in word_items.items():
            T = np.stack([it['enroll_emb'] for _, _, it in subs])
            Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
            for split, pid, it in subs:
                q = get_emb(split, pid, it)
                qn = q / (np.linalg.norm(q) + 1e-9)
                scores[pid] = float((qn @ Tn.T).max())
                y.append(it['label'])
        return roc_auc_score(y, [scores[p] for _, p, _ in sorted(items, key=lambda x: x[1])][:len(y)])

    # 直接算 (顺序需对齐)
    def mt_auc2(get_emb):
        scores = {}
        ys = []
        for w, subs in word_items.items():
            T = np.stack([it['enroll_emb'] for _, _, it in subs])
            Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
            for split, pid, it in subs:
                q = get_emb(split, pid, it)
                qn = q / (np.linalg.norm(q) + 1e-9)
                scores[pid] = float((qn @ Tn.T).max())
                ys.append(it['label'])
        return roc_auc_score(ys, [scores[pid] for _, pid, _ in items])

    def emb_orig(split, pid, it):
        return it['query_emb']

    # TTA 嵌入缓存
    tta_cache = {}

    @torch.no_grad()
    def emb_tta(split, pid, it):
        if pid in tta_cache:
            return tta_cache[pid]
        wav = str(BASE / f'val/{split}/wav/{pid}_query.wav')
        d, sr = sf.read(wav, dtype='float32')
        a = torch.from_numpy(d).float()
        if a.shape[0] > 2 * 16000:
            a = a[:2 * 16000]
        embs = []
        for sigma in [0.0, 0.005, 0.01, 0.02]:
            noisy = a + torch.randn_like(a) * sigma
            emb, _ = enc.embed_with_frames(noisy.unsqueeze(0).to("cuda"), max_sec=2.0)
            embs.append(emb[0].cpu().numpy())
        avg = np.mean(embs, axis=0)
        tta_cache[pid] = avg
        return avg

    print('原始嵌入多模板...', flush=True)
    auc0 = mt_auc2(emb_orig)
    print(f'原始: AUC={auc0:.4f}', flush=True)
    print('TTA 嵌入多模板...', flush=True)
    auc1 = mt_auc2(emb_tta)
    print(f'TTA: AUC={auc1:.4f}', flush=True)
    # 分 split
    for split in ['dev_seen', 'dev_unseen']:
        sub = [(s, p, it) for s, p, it in items if s == split]
        from collections import defaultdict as dd
        wi = dd(list)
        for s, p, it in sub:
            wi[norm(it['enroll_txt'])].append((s, p, it))
        for tag, fn in [('原始', emb_orig), ('TTA', emb_tta)]:
            sc, ys = [], []
            for w, subs in wi.items():
                T = np.stack([it['enroll_emb'] for _, _, it in subs])
                Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
                for s, p, it in subs:
                    q = fn(s, p, it)
                    qn = q / (np.linalg.norm(q) + 1e-9)
                    sc.append(float((qn @ Tn.T).max()))
                    ys.append(it['label'])
            print(f'  {split} {tag}: AUC={roc_auc_score(ys, sc):.4f}', flush=True)


if __name__ == '__main__':
    main()
