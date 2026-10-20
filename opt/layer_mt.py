"""whisper 多层嵌入多模板验证: 各 encoder 层输出的多模板 AUC"""
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
sys.path.insert(0, str(BASE))
from whisper_encoder import WhisperEncoder, log_mel_spectrogram

CKPT = BASE / "辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt"


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


@torch.no_grad()
def embed_layers(enc, audio_batch, valid_lens):
    """返回各层全局池化嵌入 (list of (B,384))"""
    mel = log_mel_spectrogram(audio_batch)  # (B,80,T)
    x = F.gelu(enc.encoder.conv1(mel))
    x = F.gelu(enc.encoder.conv2(x))
    x = x.permute(0, 2, 1)
    if x.shape[1] <= enc.encoder.positional_embedding.shape[0]:
        x = x + enc.encoder.positional_embedding[:x.shape[1]]
    else:
        x = x[:, :enc.encoder.positional_embedding.shape[0]] + enc.encoder.positional_embedding
    layers = []
    for blk in enc.encoder.blocks:
        x = blk(x)
        layers.append(x.mean(1))
    x = enc.encoder.ln_post(x)
    layers.append(x.mean(1))
    return layers  # 5 个: 4层 + ln_post


def main():
    enc = WhisperEncoder(CKPT).to("cuda")
    enc.eval()
    vd = pickle.load(open(BASE / "output_whisper/val_feats.pkl", "rb"))

    # 抽样 2000 pairs (seen+unseen 各 1000)
    import random
    random.seed(0)
    items = []
    for split in ['dev_seen', 'dev_unseen']:
        pids = sorted(vd[split].keys())
        sample = random.sample(pids, 1000)
        for pid in sample:
            items.append((split, pid, vd[split][pid]))
    print(f'抽样 {len(items)} pairs', flush=True)

    # 提取嵌入 (批量)
    layer_feats = {i: {} for i in range(5)}  # 层 -> pid -> (enroll, query)
    for i in range(0, len(items), 128):
        chunk = items[i:i + 128]
        e_batch, q_batch, e_vl, q_vl = [], [], [], []
        for split, pid, it in chunk:
            for kind, lst, vl in [('enroll', e_batch, e_vl), ('query', q_batch, q_vl)]:
                wav = str(BASE / f'val/{split}/wav/{pid}_{kind}.wav')
                d, sr = sf.read(wav, dtype='float32')
                a = torch.from_numpy(d).float()
                max_s = 2 * 16000
                if a.shape[0] > max_s:
                    a = a[:max_s]
                lst.append(a)
                vl.append(a.shape[0])
        maxl = max(max(w.shape[0] for w in e_batch), max(w.shape[0] for w in q_batch))
        e_t = torch.stack([F.pad(w, (0, maxl - w.shape[0])) for w in e_batch]).to("cuda")
        q_t = torch.stack([F.pad(w, (0, maxl - w.shape[0])) for w in q_batch]).to("cuda")
        e_l = embed_layers(enc, e_t, torch.tensor(e_vl))
        q_l = embed_layers(enc, q_t, torch.tensor(q_vl))
        for li in range(5):
            for j, (split, pid, _) in enumerate(chunk):
                layer_feats[li].setdefault(pid, {})['e'] = e_l[li][j].cpu().numpy()
                layer_feats[li][pid]['q'] = q_l[li][j].cpu().numpy()
        if (i // 128) % 5 == 0:
            print(f'  {i + len(chunk)}/{len(items)}', flush=True)

    # 各层多模板 AUC
    for split in ['dev_seen', 'dev_unseen']:
        sub = [(pid, it) for s, pid, it in items if s == split]
        from collections import defaultdict
        word_items = defaultdict(list)
        for pid, it in sub:
            word_items[norm(it['enroll_txt'])].append(pid)
        y = [it['label'] for _, it in sub]
        for li in range(5):
            scores = {}
            for w, pids in word_items.items():
                T = np.stack([layer_feats[li][p]['e'] for p in pids])
                Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
                for p in pids:
                    q = layer_feats[li][p]['q']
                    qn = q / (np.linalg.norm(q) + 1e-9)
                    scores[p] = float((qn @ Tn.T).max())
            auc = roc_auc_score(y, [scores[pid] for pid, _ in sub])
            print(f'{split} 层{li}: 多模板 AUC={auc:.4f}', flush=True)


if __name__ == '__main__':
    main()
