import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
# 验证 whisper ASR 在 30s pad 下的解码质量 (val 少量样本)
import torch, soundfile as sf, torch.nn.functional as F
from pathlib import Path
from whisper_asr import WhisperTiny, SOT, EN, TRANSCRIBE, NO_TIMESTAMPS, END, TRANSLATE, gpt2_decode
from whisper_encoder import log_mel_spectrogram

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
model = WhisperTiny(BASE / "辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt")


def asr(audio):
    pad = F.pad(audio.float(), (0, 480000 - audio.shape[0]))
    mel = log_mel_spectrogram(pad.unsqueeze(0))
    xa = model.encoder(mel)
    prompt = [SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]
    tokens = list(prompt)
    for _ in range(40):
        logits = model.decoder(torch.tensor([tokens]), xa)[0, -1]
        nxt = int(logits.argmax().item())
        if nxt == END or nxt in (SOT, TRANSLATE, TRANSCRIBE):
            break
        tokens.append(nxt)
    text = gpt2_decode([t for t in tokens[len(prompt):] if t < 50256])
    return text


if __name__ == "__main__":
    for pid in ["pair_000001", "pair_000002", "pair_000003", "pair_000004"]:
        for kind in ["enroll", "query"]:
            d, _ = sf.read(str(BASE / f"val/dev_seen/wav/{pid}_{kind}.wav"), dtype="float32")
            text = asr(torch.from_numpy(d))
            print(f"{pid}_{kind}: asr='{text}'")
        label = open(BASE / f"val/dev_seen/label/{pid}.txt", encoding="utf-8").read().strip()
        print(f"   label: {label}\n")
