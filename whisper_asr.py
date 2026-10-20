"""
Whisper ASR — 完整 tiny 模型 (encoder + decoder) 贪心解码
==========================================================
用于把 query 音频解码为 token id 序列, 与 enrollment 关键词做匹配。
特殊 token 布局 (openai whisper 20231117):
  50256 <|endoftext|>  50257 <|startoftranscript|>  50258 <|translate|>  50259 <|transcribe|>
  50260 startoflm ... 50274 startoftimestamp; 语言 token 从 50275 开始 (<|en|>=50275 ...)
  50375+ timestamp tokens
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tiktoken import get_encoding

from whisper_encoder import (AudioEncoder, LayerNorm, ResidualAttentionBlock,
                             log_mel_spectrogram, N_FFT, HOP_LENGTH, SR)

SOT = 50257
TRANSLATE = 50258
TRANSCRIBE = 50259
END = 50256
# 语言 token: whisper 官方顺序 en 在第 1 位 (89 种语言: 50275-50363)
EN = 50275
# timestamp: 50364-51863; <|notimestamps|> 是最后一个 token
NO_TIMESTAMPS = 51864

_gpt2 = None


def gpt2_decode(token_ids):
    """把 token id 序列解码为文本 (跳过特殊 id)"""
    global _gpt2
    if _gpt2 is None:
        _gpt2 = get_encoding("gpt2")
    ids = [t for t in token_ids if t < 50256]
    try:
        return _gpt2.decode(ids)
    except Exception:
        return ""


class Decoder(nn.Module):
    def __init__(self, n_vocab=51865, n_ctx=448, n_state=384, n_head=6, n_layer=4):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, cross_attention=True)
            for _ in range(n_layer)
        ])
        self.ln = LayerNorm(n_state)

    def forward(self, x, xa):
        """
        x: (1, n_tokens) token ids; xa: (1, T', n_state) encoder features
        返回: (1, n_tokens, n_vocab) logits
        """
        x = self.token_embedding(x)
        x = x + self.positional_embedding[:x.shape[1]]
        for blk in self.blocks:
            x = blk(x, xa)
        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()
        return logits


class WhisperTiny(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.encoder = AudioEncoder(n_mels=80, n_ctx=1500, n_state=384, n_head=6, n_layer=4)
        self.decoder = Decoder(n_vocab=51865, n_ctx=448, n_state=384, n_head=6, n_layer=4)
        self.load(ckpt_path)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def load(self, ckpt_path):
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['model_state_dict']
        enc_sd = {k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')}
        dec_sd = {k[len('decoder.'):]: v for k, v in sd.items() if k.startswith('decoder.')}
        self.encoder.load_state_dict(enc_sd, strict=True)
        self.decoder.load_state_dict(dec_sd, strict=True)
        print("  WhisperTiny loaded (encoder + decoder)")

    @torch.no_grad()
    def transcribe(self, audio, max_tokens=32, use_timestamps=False):
        """
        audio: (T,) @16k -> 返回 (token_ids, text)
        """
        audio = audio.float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        mel = log_mel_spectrogram(audio, n_mels=80)          # (1, 80, T')
        xa = self.encoder(mel)                                # (1, T'', 384)
        prompt = [SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]
        tokens = prompt
        for _ in range(max_tokens):
            logits = self.decoder(torch.tensor([tokens]), xa)  # (1, L, 51865)
            next_id = int(logits[0, -1].argmax().item())
            if next_id == END or next_id in (SOT, TRANSLATE, TRANSCRIBE):
                break
            tokens.append(next_id)
            if len(tokens) - len(prompt) > 24:
                break
        text = gpt2_decode(tokens[len(prompt):])
        return tokens[len(prompt):], text

    @torch.no_grad()
    def transcribe_tokens(self, audio, max_tokens=32):
        """只返回 token ids (与文本无关, 用于嵌入空间匹配)"""
        ids, _ = self.transcribe(audio, max_tokens)
        return ids


if __name__ == '__main__':
    import soundfile as sf
    from pathlib import Path
    BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
    ckpt = BASE / "辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt"
    model = WhisperTiny(ckpt)

    for pid in ['pair_000001', 'pair_000002', 'pair_000003']:
        for kind in ['enroll', 'query']:
            wav = BASE / f"val/dev_seen/wav/{pid}_{kind}.wav"
            d, sr = sf.read(str(wav), dtype='float32')
            ids, text = model.transcribe(torch.from_numpy(d))
            print(f"{pid}_{kind}: asr_text='{text}' ids={ids[:8]}")
        label = open(BASE / f"val/dev_seen/label/{pid}.txt", encoding='utf-8').read().strip()
        print(f"    label: {label}\n")
