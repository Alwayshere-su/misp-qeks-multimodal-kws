"""whisper base/small 嵌入的多模板验证 (val 抽样) — 更强嵌入模型"""
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


@torch.no_grad()
def embed_whisper(model, audio_batch, valid_lens, max_sec=2.0):
    """官方 whisper encoder 嵌入, 有效帧掩码平均"""
    import whisper
    mels = []
    for a, vl in zip(audio_batch, valid_lens):
        a = a[:int(max_sec * 16000)]
        m = whisper.log_mel_spectrogram(whisper.pad_or_trim(a))
        mels.append(m)
    mel_batch = torch.stack(mels).to(model.device)
    feats = model.embed_audio(mel_batch)  # (B, T', dim)
    # 有效帧掩码
    n_frames = feats.shape[1]
    n_valid = ((torch.tensor(valid_lens) // 160) // 2).clamp(1, n_frames).to(model.device)
    mask = (torch.arange(n_frames, device=model.device).unsqueeze(0) < n_valid.unsqueeze(1)).float().unsqueeze(-1)
    emb = (feats * mask).sum(1) / mask.sum(1).clamp(min=1)
    return emb.cpu().numpy()


def main():
    import whisper
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'base'
    model = whisper.load_model(model_name)
    model.eval()
    print(f'model: {model_name}, state: {model.dims.n_audio_state}', flush=True)

    vd = pickle.load(open(BASE / "output_whisper/val_feats.pkl", "rb"))
    import random
    random.seed(0)
    items = []
    for split in ['dev_seen', 'dev_unseen']:
        pids = random.sample(sorted(vd[split].keys()), 800)
        for pid in pids:
            items.append((split, pid, vd[split][pid]))
    print(f'抽样 {len(items)}', flush=True)

    # 提取嵌入
    e_emb = {}
    q_emb = {}
    for i in range(0, len(items), 64):
        chunk = items[i:i + 64]
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
        if (i // 64) % 5 == 0:
            print(f'  {i + len(chunk)}/{len(items)}', flush=True)

    # 多模板 AUC
    from collections import defaultdict
    word_items = defaultdict(list)
    for split, pid, it in items:
        word_items[norm(it['enroll_txt'])].append((split, pid, it))
    for split in ['dev_seen', 'dev_unseen']:
        sub = [(s, p, it) for s, p, it in items if s == split]
        wi = defaultdict(list)
        for s, p, it in sub:
            wi[norm(it['enroll_txt'])].append((s, p, it))
        sc, ys = [], []
        for w, subs in wi.items():
            T = np.stack([e_emb[p] for _, p, _ in subs])
            Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
            for s, p, it in subs:
                q = q_emb[p]
                qn = q / (np.linalg.norm(q) + 1e-9)
                sc.append(float((qn @ Tn.T).max()))
                ys.append(it['label'])
        print(f'{split}: {model_name} 多模板 AUC={roc_auc_score(ys, sc):.4f}', flush=True)


if __name__ == '__main__':
    main()
