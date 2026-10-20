"""
评估 Whisper tiny ASR 转写质量 (val 10K query + 部分 enroll)
目标: 文本匹配 AUC 能否接近 1.0
"""
import torch, soundfile as sf, pickle, time, sys
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from whisper_asr import WhisperTiny
from whisper_encoder import log_mel_spectrogram, SR

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
CKPT = BASE / "辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WhisperTiny(CKPT).to(DEVICE)
print(f"device={DEVICE}")

PAD_SEC = 30  # whisper 训练用 30s, pad 避免位置编码问题
PAD_LEN = PAD_SEC * SR

def asr_text(audio, max_tokens=40):
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[1] < PAD_LEN:
        audio = torch.nn.functional.pad(audio, (0, PAD_LEN - audio.shape[1]))
    mel = log_mel_spectrogram(audio)  # (1,80,T')
    xa = model.encoder(mel)
    from whisper_asr import SOT, EN, TRANSCRIBE, NO_TIMESTAMPS, END, TRANSLATE, gpt2_decode
    tokens = [SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]
    for _ in range(max_tokens):
        logits = model.decoder(torch.tensor([tokens], device=DEVICE), xa)[0, -1]
        nxt = int(logits.argmax().item())
        if nxt == END or nxt in (SOT, TRANSLATE, TRANSCRIBE):
            break
        tokens.append(nxt)
    text = gpt2_decode([t for t in tokens[4:] if t < 50256])
    return text

def text_match_score(et, qt):
    if not et or not qt:
        return 0.5
    et = et.lower().strip(); qt = qt.lower().strip()
    if et in qt:
        return 1.0
    ewords = set(w for w in et.split() if len(w) > 2)
    qwords = set(qt.split())
    if ewords & qwords:
        return 0.9
    return 0.0

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='0=全部')
    args = parser.parse_args()

    d = pickle.load(open(BASE / "output_whisper/val_feats.pkl", 'rb'))
    items = []
    for name in ['dev_seen', 'dev_unseen']:
        for pid, it in d[name].items():
            items.append((name, pid, it))
    if args.limit:
        items = items[:args.limit]

    # 预加载真实 query_txt 用于评估
    t0 = time.time()
    y_true, y_pred, exact_ok = [], [], [0, 0]
    n = 0
    for name, pid, it in tqdm(items, desc="ASR"):
        wav = BASE / f"val/{name}/wav/{pid}_query.wav"
        audio, sr = sf.read(str(wav), dtype='float32')
        audio = torch.from_numpy(audio).float().to(DEVICE)
        text = asr_text(audio)
        et = it['enroll_txt']; qt_true = it['query_txt']
        score = text_match_score(et, text)
        y_true.append(it['label'])
        y_pred.append(score)
        # 转写准确率 (query_txt 是否在 ASR 输出中)
        if it['label'] == 1:
            exact_ok[0] += 1 if qt_true.lower() in text.lower() else 0
        else:
            exact_ok[1] += 1 if qt_true.lower() in text.lower() else 0
        n += 1

    auc = roc_auc_score(y_true, y_pred)
    print(f"\n=== ASR 文本匹配评估 (n={n}, {time.time()-t0:.0f}s) ===")
    print(f"AUC = {auc:.6f}")
    print(f"pos exact-in-query: {exact_ok[0]}/{sum(y_true)}")
    print(f"neg exact-in-query: {exact_ok[1]}/{len(y_true)-sum(y_true)}")

if __name__ == '__main__':
    main()
