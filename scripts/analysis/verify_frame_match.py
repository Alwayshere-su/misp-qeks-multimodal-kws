"""
快速验证: whisper 帧级匹配是否优于全局 cosine
===============================================
对 val 10K pairs 计算:
  1) 全局 cosine AUC (baseline)
  2) 帧级匹配相似度 AUC (enroll 帧 -> query 帧的最近邻平均)
"""
import pickle, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from whisper_encoder import WhisperEncoder
from extract_whisper_feats import load_audio

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
VAL = BASE / "val"
CKPT = BASE / "辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 128


@torch.no_grad()
def extract_frames_batch(enc, paths, max_sec, desc):
    outs = {}
    for i in tqdm(range(0, len(paths), BATCH), desc=desc, leave=False):
        chunk = paths[i:i + BATCH]
        waves, valid_lens = [], []
        for p in chunk:
            w, v = load_audio(p, max_sec)
            waves.append(w); valid_lens.append(v)
        maxl = max(w.shape[0] for w in waves)
        batch = torch.stack([F.pad(w, (0, maxl - w.shape[0])) for w in waves]).to(DEVICE)
        vl = torch.tensor(valid_lens, dtype=torch.long)
        _, frames = enc.embed_with_frames(batch, valid_lens=vl, n_frames=24)  # (B, 24, 384)
        for p, fr in zip(chunk, frames.cpu().numpy()):
            outs[str(p)] = fr
    return outs


def main():
    enc = WhisperEncoder(CKPT).to(DEVICE)
    enc.eval()
    print(f"Device: {DEVICE}")

    from sklearn.metrics import roc_auc_score
    all_cos_g, all_cos_f, all_y = [], [], []
    for name in ['dev_seen', 'dev_unseen']:
        d = Path(VAL) / name
        wav_dir = d / "wav"
        enroll_paths = sorted(wav_dir.glob("pair_*_enroll.wav"))
        query_paths = sorted(wav_dir.glob("pair_*_query.wav"))
        e_fr = extract_frames_batch(enc, enroll_paths, 1.5, f"{name} enroll")
        q_fr = extract_frames_batch(enc, query_paths, 2.0, f"{name} query")

        for ep in enroll_paths:
            pid = ep.stem.replace('_enroll', '')
            qp = wav_dir / f"{pid}_query.wav"
            parts = open(d / "label" / f"{pid}.txt", encoding='utf-8').read().strip().split('\t')
            y = int(parts[2].replace('label:', ''))
            ef = torch.tensor(e_fr[str(ep)], dtype=torch.float32)  # (24, 384)
            qf = torch.tensor(q_fr[str(qp)], dtype=torch.float32)
            # 全局
            eg = ef.mean(0); qg = qf.mean(0)
            eg = eg / (eg.norm() + 1e-9); qg = qg / (qg.norm() + 1e-9)
            # 帧级: 每 enroll 帧 -> query 最近帧 cosine 的均值
            ef_n = ef / (ef.norm(dim=-1, keepdim=True) + 1e-9)
            qf_n = qf / (qf.norm(dim=-1, keepdim=True) + 1e-9)
            sim = ef_n @ qf_n.T          # (24, 24)
            frame_score = sim.max(dim=1).values.mean().item()
            all_cos_g.append((eg * qg).sum().item())
            all_cos_f.append(frame_score)
            all_y.append(y)

        g_auc = roc_auc_score(all_y, all_cos_g)
        f_auc = roc_auc_score(all_y, all_cos_f)
        print(f"{name}: global cosine AUC={g_auc:.4f} | frame-match AUC={f_auc:.4f}")

    g_auc = roc_auc_score(all_y, all_cos_g)
    f_auc = roc_auc_score(all_y, all_cos_f)
    print(f"\nALL: global={g_auc:.4f} frame={f_auc:.4f}")


if __name__ == '__main__':
    main()
