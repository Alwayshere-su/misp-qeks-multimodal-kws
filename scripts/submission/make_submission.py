import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
增量提取 + 预测 — 带断点续传，防止长任务被 kill
"""
import os, pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import soundfile as sf
import torchaudio

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
TEST = BASE / "test"
OUT = BASE / "output_v3"
SUB = BASE / "submission"
SUB.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000

# ============================================================
# Model (exact copy from baseline_v3.py)
# ============================================================
class AudioEncoderV3(nn.Module):
    def __init__(self, in_c=120, hidden=256, out_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_c, hidden, 7, 2, 3), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, 2, 2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, 2, 2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, 2, 2), nn.BatchNorm1d(hidden), nn.GELU(),
        )
        self.attn = nn.Conv1d(hidden, 1, 1)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.encoder(x)
        w = self.attn(x).softmax(-1)
        global_e = (w * x).sum(-1)
        global_e = self.proj(global_e)
        frames = x.transpose(1, 2)
        frames = self.proj(frames)
        return global_e, frames

class KWSpotterV3(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.encoder = AudioEncoderV3(120, hidden, hidden)
        self.match_attn = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=0.1)
        self.match_norm = nn.LayerNorm(hidden)
        self.cls = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden * 2, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, ef, qf):
        e_global, e_frames = self.encoder(ef)
        q_global, q_frames = self.encoder(qf)
        q_attn, _ = self.match_attn(e_global.unsqueeze(1), q_frames, q_frames)
        q_attn = self.match_norm(q_attn.squeeze(1))
        diff = torch.abs(e_global - q_attn)
        prod = e_global * q_attn
        fused = torch.cat([e_global, q_attn, diff, prod], dim=-1)
        return self.cls(fused)

    def predict(self, ef, qf):
        with torch.no_grad():
            return torch.sigmoid(self.forward(ef, qf)).squeeze(-1)

# ============================================================
# Feature extraction
# ============================================================

def extract_mfcc_feats(wav_path, max_sec):
    data, sr = sf.read(str(wav_path), dtype='float32')
    if data.ndim > 1: data = data.mean(axis=1)
    audio = torch.from_numpy(data).float()
    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
    max_s = int(max_sec * SR)
    if len(audio) > max_s: audio = audio[:max_s]
    elif len(audio) < max_s: audio = F.pad(audio, (0, max_s - len(audio)))
    audio = audio.unsqueeze(0)
    mfcc = torchaudio.transforms.MFCC(SR, n_mfcc=40, melkwargs={
        'n_mels': 80, 'n_fft': 512, 'hop_length': 160, 'win_length': 400
    })(audio)
    delta = torchaudio.transforms.ComputeDeltas()
    d1, d2 = delta(mfcc), delta(delta(mfcc))
    feat = torch.cat([mfcc, d1, d2], dim=1)
    return feat.squeeze(0)

def extract_with_checkpoint(split_type, checkpoint_every=5000):
    """增量提取 + 定期保存"""
    if split_type == 'inset':
        wav_dir = TEST / "eval_seen_unlabeled" / "wav"
        cache_path = OUT / "test_inset_feats.pkl"
    else:
        wav_dir = TEST / "eval_unseen_unlabeled" / "wav"
        cache_path = OUT / "test_outset_feats.pkl"

    # Resume from checkpoint
    ckpt_path = OUT / f"test_{split_type}_ckpt.pkl"
    if ckpt_path.exists():
        with open(ckpt_path, 'rb') as f:
            data, processed = pickle.load(f)
        print(f"  Resuming from {len(processed)} already processed")
    else:
        data, processed = {}, set()

    files = sorted(wav_dir.glob("pair_*_enroll.wav"))
    remaining = [(f, f.stem.replace('_enroll', '')) for f in files if f.stem.replace('_enroll', '') not in processed]
    print(f"  Total: {len(files)}, Already done: {len(processed)}, Remaining: {len(remaining)}")

    if not remaining:
        return data

    for i, (f, pid) in enumerate(tqdm(remaining, desc=f"  {split_type}")):
        try:
            ef = extract_mfcc_feats(wav_dir / f"{pid}_enroll.wav", 2.0)
            qf = extract_mfcc_feats(wav_dir / f"{pid}_query.wav", 4.0)
            data[pid] = {'enroll_feat': ef, 'query_feat': qf}
            processed.add(pid)
        except Exception as e:
            print(f"  skip {pid}: {e}")
            data[pid] = {'enroll_feat': torch.zeros(120, 100), 'query_feat': torch.zeros(120, 200)}
            processed.add(pid)

        # Save checkpoint
        if (i + 1) % checkpoint_every == 0:
            with open(ckpt_path, 'wb') as f:
                pickle.dump((data, processed), f)
            print(f"  [ckpt] {len(data)} pairs saved")

    # Final save
    print(f"  Saving final cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

    # Clean checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()

    return data

# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict_test(model, test_data, desc="Predict"):
    model.eval()
    preds = {}
    items = list(test_data.items())
    for pid, d in tqdm(items, desc=f"  {desc}"):
        ef = d['enroll_feat'].unsqueeze(0).to(DEVICE)
        qf = d['query_feat'].unsqueeze(0).to(DEVICE)
        prob = model.predict(ef, qf).cpu().item()
        preds[pid] = prob
    return preds

def make_csv(inset_preds, outset_preds, path):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},inset,{inset_preds.get(pid, 0.5):.6f}\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},outset,{outset_preds.get(pid, 0.5):.6f}\n")
    print(f"  -> {path}")

# ============================================================
# Main
# ============================================================

def main():
    print(f"Device: {DEVICE}")

    # 1. Extract outset features (with checkpoint resume)
    print("\n[1] Extracting outset features (checkpoint every 5000)...")
    outset_data = extract_with_checkpoint("outset", checkpoint_every=5000)

    # 2. Load inset features
    print("\n[2] Loading inset features...")
    inset_cache = OUT / "test_inset_feats.pkl"
    if inset_cache.exists():
        with open(inset_cache, 'rb') as f:
            inset_data = pickle.load(f)
        print(f"  Loaded {len(inset_data)} pairs")
    else:
        print("  Extracting...")
        inset_data = extract_with_checkpoint("inset", checkpoint_every=5000)

    # 3. Load model
    print("\n[3] Loading model (wide, hidden=320)...")
    model = KWSpotterV3(hidden=320).to(DEVICE)
    model.load_state_dict(torch.load(OUT / "best_model_v3.pt", weights_only=True))
    model.eval()
    print("  OK")

    # 4. Predict
    print("\n[4] Predicting...")
    inset_preds = predict_test(model, inset_data, "inset")
    outset_preds = predict_test(model, outset_data, "outset")

    # 5. Generate submission
    print("\n[5] Generating submission...")
    out_path = SUB / "submit_v3_auc0.6466.csv"
    make_csv(inset_preds, outset_preds, out_path)

    print(f"\n  Done! -> {out_path}")

if __name__ == "__main__":
    main()
