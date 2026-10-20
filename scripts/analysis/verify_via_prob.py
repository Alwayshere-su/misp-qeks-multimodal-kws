"""实验: whisper 条件解码概率做关键词验证 vs 贪心 ASR 文本匹配"""
import re
import torch
import torch.nn.functional as F
import whisper
import whisper.tokenizer as wt
import soundfile as sf
from pathlib import Path
from sklearn.metrics import roc_auc_score

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9' ]", '', (s or '').lower()).strip()


def main():
    model_name = 'small'
    model = whisper.load_model(model_name)
    tok = wt.get_tokenizer(multilingual=True, language='en', task='transcribe')
    prompt = tok.sot_sequence_including_notimestamps
    print('prompt:', prompt)

    # 加载样本: dev_unseen 前 300 (150 正 + 150 负)
    labels = {}
    for f in (BASE / 'val/dev_unseen/label').glob('*.txt'):
        parts = f.read_text(encoding='utf-8').strip().split('\t')
        labels[f.stem] = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in parts}
    items = [(pid, d) for pid, d in labels.items() if d['label'] == '1'][:150]
    items += [(pid, d) for pid, d in labels.items() if d['label'] == '0'][:150]

    opt = whisper.DecodingOptions(language='en', without_timestamps=True, fp16=False)
    y, s_asr, s_prob = [], [], []
    n_token_fail = 0
    for pid, d in items:
        wav = str(BASE / f'val/dev_unseen/wav/{pid}_query.wav')
        a, sr = sf.read(wav, dtype='float32')
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(torch.from_numpy(a))).to(model.device)
        # A: 贪心 ASR
        res = whisper.decode(model, mel, opt)
        e = norm(d['enrollment'])
        q = norm(res.text)
        s_asr.append(1.0 if (e == q or e in q or q in e) else 0.0)
        # B: 条件解码概率
        try:
            ids = tok.encode(d['enrollment'])
        except Exception:
            ids = []
        if not ids:
            s_prob.append(0.5)
            n_token_fail += 1
            continue
        af = model.embed_audio(mel.unsqueeze(0))
        tokens = list(prompt) + ids
        with torch.no_grad():
            logits = model.decoder(torch.tensor([tokens], device=model.device), af)[0]
        logp = F.log_softmax(logits.float(), dim=-1)
        probs = [logp[len(prompt) + i, ids[i]].item() for i in range(len(ids))]
        s_prob.append(sum(probs) / len(probs))
        y.append(1 if d['label'] == '1' else 0)
    print(f'已处理 {len(y)} 样本, token 编码失败 {n_token_fail}')
    auc_asr = roc_auc_score(y, s_asr)
    auc_prob = roc_auc_score(y, s_prob)
    print(f'[small dev_unseen 300] ASR贪心AUC={auc_asr:.6f}  条件概率AUC={auc_prob:.6f}')
    # 融合
    import numpy as np
    sa = np.array(s_asr); sp = np.array(s_prob)
    for w in [0.3, 0.5, 0.7]:
        sc = w * sa + (1 - w) * (sp - sp.min()) / (sp.max() - sp.min() + 1e-9)
        print(f'  融合 w={w}: AUC={roc_auc_score(y, sc):.6f}')


if __name__ == '__main__':
    main()
