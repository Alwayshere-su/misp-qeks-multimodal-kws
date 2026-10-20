"""
LipReading 视频特征批量提取 + 价值评估
=======================================
提取 val/test 的 enroll/query 视频嵌入(256维), 批量 GPU 处理。
"""
import pickle, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from lipreading_encoder import LipReadingEncoder, extract_video_frames

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
CKPT = BASE / "辅助资源/pretrained_feature_models/lipreading/lipreading_LRW_0.8018.pt"
OUT = BASE / "output_lip"
OUT.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 24
BATCH = 32


@torch.no_grad()
def extract_mp4s(enc, mp4_paths, desc):
    """返回 {str(path): emb(256,)}"""
    out = {}
    for i in tqdm(range(0, len(mp4_paths), BATCH), desc=desc, leave=False):
        chunk = mp4_paths[i:i + BATCH]
        frames_list = []
        for p in chunk:
            try:
                f = extract_video_frames(p, max_frames=MAX_FRAMES)
            except Exception:
                f = torch.zeros(1, MAX_FRAMES, 96, 96)
            frames_list.append(f)
        batch = torch.stack(frames_list).to(DEVICE)          # (B, 1, T, 96, 96)
        emb = enc.embed(batch)                                # (B, 256)
        for p, e in zip(chunk, emb.cpu().numpy()):
            out[str(p)] = e
    return out


def extract_split(enc, split_dir, split_name, is_test=False, text_map=None):
    d = Path(split_dir)
    mp4_dir = d / "mp4"
    enroll_paths = sorted(mp4_dir.glob("pair_*_enroll.mp4"))
    query_paths = sorted(mp4_dir.glob("pair_*_query.mp4"))
    print(f"  {split_name}: {len(enroll_paths)} pairs")

    e_embs = extract_mp4s(enc, enroll_paths, f"{split_name} enroll")
    q_embs = extract_mp4s(enc, query_paths, f"{split_name} query")

    data = {}
    for ep in enroll_paths:
        pid = ep.stem.replace('_enroll', '')
        qp = mp4_dir / f"{pid}_query.mp4"
        item = {'enroll_video': e_embs[str(ep)], 'query_video': q_embs[str(qp)]}
        if not is_test:
            parts = open(d / "label" / f"{pid}.txt", encoding='utf-8').read().strip().split('\t')
            item['label'] = int(parts[2].replace('label:', ''))
        data[pid] = item
    return data


def load_test_texts():
    res = {}
    for key, path in [('inset', TEST / "eval_seen_enroll_text.csv"),
                      ('outset', TEST / "eval_unseen_enroll_text.csv")]:
        m = {}
        with open(path, encoding='utf-8') as f:
            next(f)
            for line in f:
                pid, txt = line.strip().split(',', 1)
                m[pid] = txt
        res[key] = m
    return res


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-only', action='store_true')
    args = parser.parse_args()

    global TEST
    TEST = BASE / "test"
    t0 = time.time()
    enc = LipReadingEncoder(CKPT).to(DEVICE)
    enc.eval()
    print(f"Device: {DEVICE}")

    print("\n[1] val 视频特征提取...")
    val_pkl = OUT / "val_lip.pkl"
    if val_pkl.exists():
        with open(val_pkl, 'rb') as f:
            val_data = pickle.load(f)
        print(f"  loaded existing val_lip.pkl")
    else:
        val_data = {}
        for name in ['dev_seen', 'dev_unseen']:
            val_data[name] = extract_split(enc, BASE / "val" / name, f"val/{name}")
        with open(val_pkl, 'wb') as f:
            pickle.dump(val_data, f)
        print(f"  saved val_lip.pkl ({time.time()-t0:.0f}s)")

    # 评估视频 cosine AUC
    from sklearn.metrics import roc_auc_score
    for name in ['dev_seen', 'dev_unseen']:
        items = list(val_data[name].items())
        E = np.stack([it['enroll_video'] for _, it in items])
        Q = np.stack([it['query_video'] for _, it in items])
        y = np.array([it['label'] for _, it in items])
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        cos = (En * Qn).sum(1)
        auc = roc_auc_score(y, cos)
        print(f"  >>> {name} video cosine AUC = {auc:.4f} (whisper audio global: "
              f"{'0.8179' if name=='dev_seen' else '0.9558'})")

    if args.val_only:
        print(f"\n[val-only] done ({time.time()-t0:.0f}s)")
        return

    print("\n[2] test 视频特征提取...")
    text_maps = load_test_texts()
    for key, name in [('inset', 'eval_seen_unlabeled'), ('outset', 'eval_unseen_unlabeled')]:
        test_data = extract_split(enc, TEST / name, f"test/{key}", is_test=True)
        with open(OUT / f"test_lip_{key}.pkl", 'wb') as f:
            pickle.dump(test_data, f)
        print(f"  saved test_lip_{key}.pkl ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
