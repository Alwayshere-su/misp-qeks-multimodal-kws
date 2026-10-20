"""Proper whisper decoding: temperature fallback + repetition penalty (mimic openai/whisper)."""
import sys, pickle, random
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
sys.path.insert(0, str(BASE))
from whisper_asr import WhisperTiny, SOT, EN, TRANSCRIBE, NO_TIMESTAMPS, END, gpt2_decode
from whisper_encoder import log_mel_spectrogram, SR

CKPT = BASE / "辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WhisperTiny(CKPT).to(DEVICE)
PAD_LEN = 30 * SR

# tokens to always suppress (openai default suppress_tokens minus a few)
DEFAULT_SUPPRESS = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62,
                    63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922,
                    931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846,
                    3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929,
                    11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618,
                    16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161,
                    26435, 28279, 29464, 31650, 32302, 32470, 36865, 38563, 39327,
                    40236, 44611, 48819, 49880, 50358, 50360, 50361, 50362]


def detect_repetition(text, max_len=16):
    """whisper 的 repetition detection: 重复的 n-gram 占比"""
    if len(text) < 8:
        return False
    n = len(text)
    for i in range(3, 7):
        if n // i < 3:
            break
        seen = set()
        reps = 0
        for j in range(0, n - i + 1):
            gram = text[j:j + i]
            if gram in seen:
                reps += 1
            seen.add(gram)
        if reps > 0.4 * (n - i + 1):
            return True
    return False


@torch.no_grad()
def asr_proper(audio, max_tokens=40, temps=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[1] < PAD_LEN:
        audio = torch.nn.functional.pad(audio, (0, PAD_LEN - audio.shape[1]))
    mel = log_mel_spectrogram(audio)
    xa = model.encoder(mel)
    prompt = [SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]

    for temp in temps:
        tokens = list(prompt)
        for _ in range(max_tokens):
            logits = model.decoder(torch.tensor([tokens], device=DEVICE), xa)[0, -1]
            logits[torch.tensor(DEFAULT_SUPPRESS, device=DEVICE)] = -float("inf")
            if temp == 0:
                nxt = int(logits.argmax().item())
            else:
                probs = torch.softmax(logits / temp, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            if nxt == END:
                break
            tokens.append(nxt)
            if len(tokens) - len(prompt) >= max_tokens:
                break
        text = gpt2_decode([t for t in tokens[len(prompt):] if t < 50256])
        # 官方接受条件: 含结束符 或 无重复
        if not detect_repetition(text):
            return text
    return text


def fuzzy_score(et, qt):
    et = et.lower().strip(); qt = qt.lower().strip()
    if not et or not qt:
        return 0.0
    if et in qt:
        return 1.0
    best = 0.0
    n = len(et)
    for i in range(max(1, len(qt) - n + 1)):
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
        text = asr_proper(audio)
        et = it["enroll_txt"]; qt = it["query_txt"]
        y.append(it["label"])
        s.append(fuzzy_score(et, text))
        if it["label"] == 1:
            print(f"[pos] enroll='{et}' | true='{qt}' | asr='{text}' | fz={s[-1]:.2f}")
        elif random.random() < 0.08:
            print(f"[neg] enroll='{et}' | true='{qt}' | asr='{text}' | fz={s[-1]:.2f}")

    y = np.array(y); s = np.array(s)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    print(f"\nfuzzy-match AUC (n={len(y)}): {auc:.4f}")


if __name__ == "__main__":
    main()
