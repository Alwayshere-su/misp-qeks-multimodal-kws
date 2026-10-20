"""验证 test 与 val 音频的特征分布偏移 (whisper encoder 嵌入)"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from whisper_encoder import WhisperEncoder
import soundfile as sf
import torch

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')
CKPT = BASE / "辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt"


def main():
    enc = WhisperEncoder(CKPT)
    emb_dim = 384
    # 抽样嵌入
    import random
    random.seed(0)
    stats = {}
    for tag, folder, n in [
        ('val_seen_q', 'val/dev_seen/wav', 300),
        ('val_unseen_q', 'val/dev_unseen/wav', 300),
        ('test_seen_q', 'test/eval_seen_unlabeled/wav', 300),
        ('test_unseen_q', 'test/eval_unseen_unlabeled/wav', 300),
    ]:
        files = sorted((BASE / folder).glob('*_query.wav'))
        sample = random.sample(files, min(n, len(files)))
        embs = []
        for f in sample:
            d, sr = sf.read(str(f), dtype='float32')
            with torch.no_grad():
                emb = enc.embed(torch.from_numpy(d), max_sec=2.0).cpu().numpy()[0]
            embs.append(emb)
        embs = np.array(embs)
        stats[tag] = embs
        print(f'{tag}: n={len(embs)} norm_mean={np.linalg.norm(embs, axis=1).mean():.3f} '
              f'mean_vec_norm={np.linalg.norm(embs.mean(0)):.3f}', flush=True)
    # 跨集余弦相似度 (质心)
    for a, b in [('val_seen_q', 'test_seen_q'), ('val_unseen_q', 'test_unseen_q')]:
        ca = stats[a].mean(0); cb = stats[b].mean(0)
        cos = np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-9)
        print(f'{a} vs {b} 质心余弦: {cos:.4f}', flush=True)


if __name__ == '__main__':
    main()
