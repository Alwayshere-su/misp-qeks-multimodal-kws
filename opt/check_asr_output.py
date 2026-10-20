"""Inspect whisper-tiny ASR outputs vs ground truth query texts on val."""
import pickle, sys, random
from pathlib import Path
import torch
import soundfile as sf

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
sys.path.insert(0, str(BASE))
from whisper_asr import WhisperTiny
from whisper_encoder import log_mel_spectrogram, SR

CKPT = BASE / "辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WhisperTiny(CKPT).to(DEVICE)
PAD_LEN = 30 * SR


def asr_text(audio, max_tokens=40):
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[1] < PAD_LEN:
        audio = torch.nn.functional.pad(audio, (0, PAD_LEN - audio.shape[1]))
    mel = log_mel_spectrogram(audio)
    xa = model.encoder(mel)
    from whisper_asr import SOT, EN, TRANSCRIBE, NO_TIMESTAMPS, END, gpt2_decode
    tokens = [SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]
    for _ in range(max_tokens):
        logits = model.decoder(torch.tensor([tokens], device=DEVICE), xa)[0, -1]
        nxt = int(logits.argmax().item())
        if nxt == END:
            break
        tokens.append(nxt)
    return gpt2_decode([t for t in tokens[4:] if t < 50256])


def fuzzy_score(et, qt):
    """edit-distance based containment score"""
    et = et.lower().strip(); qt = qt.lower().strip()
    if not et or not qt:
        return 0.0
    if et in qt:
        return 1.0
    # sliding window edit distance
    best = 0.0
    n = len(et)
    for i in range(len(qt) - n + 1):
        window = qt[i:i + n]
        dist = sum(1 for a, b in zip(et, window) if a != b) / n
        best = max(best, 1.0 - dist)
    return best


def main():
    d = pickle.load(open(BASE / "output_whisper/val_feats.pkl", "rb"))
    items = []
    for name in ["dev_seen", "dev_unseen"]:
        for pid, it in d[name].items():
            items.append((name, pid, it))
    random.seed(0)
    random.shuffle(items)
    sample = items[:120]

    y, s = [], []
    for name, pid, it in sample:
        wav = BASE / f"val/{name}/wav/{pid}_query.wav"
        audio, sr_ = sf.read(str(wav), dtype="float32")
        audio = torch.from_numpy(audio).float().to(DEVICE)
        text = asr_text(audio)
        et = it["enroll_txt"]; qt = it["query_txt"]
        y.append(it["label"])
        s.append(fuzzy_score(et, text))
        if it["label"] == 1:
            print(f"[pos] enroll='{et}' | true='{qt}' | asr='{text}' | fz={s[-1]:.2f}")
        elif random.random() < 0.1:
            print(f"[neg] enroll='{et}' | true='{qt}' | asr='{text}' | fz={s[-1]:.2f}")

    y = torch.tensor(y)
    s = torch.tensor(s)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    order = torch.argsort(s)
    ranks = torch.empty(len(s)); ranks[order] = torch.arange(1, len(s) + 1)
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    print(f"\nfuzzy-match AUC (n={len(y)}): {auc:.4f}")


if __name__ == "__main__":
    main()
