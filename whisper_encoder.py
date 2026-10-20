"""
OpenAI Whisper Encoder — 手写官方架构, 直接加载 tiny.pt state_dict
====================================================================
只使用 encoder 部分提取音频嵌入 (不需要 decoder/ASR)。
特征: log-mel(80) → conv1+conv2 → 4层transformer → ln_post → 帧嵌入 → 全局池化

参考: https://github.com/openai/whisper/blob/main/whisper/model.py / audio.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
SR = 16000

# ------------------------------------------------------------
# log-mel 频谱 (复刻 openai/whisper/audio.py; mel filter 手写, 与 librosa 一致)
# ------------------------------------------------------------

_mel_filter_cache = {}

def hz_to_mel(freq, htk=False):
    if htk:
        return 2595.0 * np.log10(1.0 + freq / 700.0)
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (freq - f_min) / f_sp
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    mels = np.where(freq >= min_log_hz, min_log_mel + np.log(freq / min_log_hz) / logstep, mels)
    return mels


def mel_to_hz(mels, htk=False):
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    freqs = np.where(mels >= min_log_mel, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs)
    return freqs


def mel_frequencies(n_mels, fmin=0.0, fmax=11025.0, htk=False):
    return mel_to_hz(np.linspace(hz_to_mel(fmin, htk), hz_to_mel(fmax, htk), n_mels), htk)


def get_mel_filters(n_mels=80, sr=SR, n_fft=N_FFT):
    """与 librosa.filters.mel(sr, n_fft, n_mels, norm='slaney') 一致 (openai whisper 同款)"""
    key = (n_mels, sr, n_fft)
    if key not in _mel_filter_cache:
        fmax = float(sr) / 2
        weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
        fftfreqs = np.linspace(0, float(sr) / 2, int(1 + n_fft // 2))
        mel_f = mel_frequencies(n_mels + 2, 0.0, fmax, False)
        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)
        for i in range(n_mels):
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]
            weights[i] = np.maximum(0, np.minimum(lower, upper))
        # slaney 归一化
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
        _mel_filter_cache[key] = torch.from_numpy(weights)
    return _mel_filter_cache[key]


def log_mel_spectrogram(audio, n_mels=80, padding=0):
    """
    audio: (T,) float32 @16kHz (或 (B, T))
    返回: (B, n_mels, T') 或 (n_mels, T')
    与 whisper 一致: log10 clamp + 归一化到 [-1.5, ~1]
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    window = torch.hann_window(WIN_LENGTH, device=audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, WIN_LENGTH, window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = get_mel_filters(n_mels).to(audio.device)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec  # (B, n_mels, T')


# ------------------------------------------------------------
# Whisper Encoder (仅编码器部分)
# ------------------------------------------------------------

class LayerNorm(nn.Module):
    """openai whisper 命名: weight/bias"""
    def __init__(self, n_state, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_state))
        self.bias = nn.Parameter(torch.zeros(n_state))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state, n_head):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)  # openai 的 key 无 bias
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(self, x, xa=None, mask=None):
        q = self.query(x)
        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)
        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q, k, v, mask=None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        qk = q @ k
        if mask is not None:
            qk = qk + mask
        qk = qk.float().softmax(dim=-1).to(q.dtype)
        return (qk @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, cross_attention=False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x, xa=None, mask=None):
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    """whisper 编码器: (B, 80, T) log-mel -> (B, T', n_state) 帧嵌入"""
    def __init__(self, n_mels=80, n_ctx=1500, n_state=384, n_head=6, n_layer=4):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, 3, 1, 1)
        self.conv2 = nn.Conv1d(n_state, n_state, 3, 2, 1)
        self.register_buffer("positional_embedding", torch.zeros(n_ctx, n_state))
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x):
        """
        x: (B, 80, T) log-mel
        返回: (B, T', n_state)
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T', n_state)
        T = x.shape[1]
        if T <= self.positional_embedding.shape[0]:
            x = x + self.positional_embedding[:T]
        else:
            # 超长输入: 截断 (不应对本任务发生)
            x = x[:, :self.positional_embedding.shape[0]] + self.positional_embedding
        for blk in self.blocks:
            x = blk(x)
        return self.ln_post(x)


class WhisperEncoder(nn.Module):
    """完整封装: 音频 -> 全局嵌入"""
    def __init__(self, ckpt_path, n_mels=80, pool='mean'):
        super().__init__()
        self.encoder = AudioEncoder(
            n_mels=n_mels, n_ctx=1500, n_state=384, n_head=6, n_layer=4)
        self.pool = pool
        self.load_from_ckpt(ckpt_path)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def load_from_ckpt(self, ckpt_path):
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        full = sd['model_state_dict']
        our = {}
        for k, v in full.items():
            if k.startswith('encoder.'):
                our[k[len('encoder.'):]] = v
        missing, unexpected = self.encoder.load_state_dict(our, strict=True)
        print(f"  WhisperEncoder loaded from {ckpt_path}")
        print(f"  encoder params: {sum(p.numel() for p in self.encoder.parameters())/1e6:.2f}M")

    @torch.no_grad()
    def embed(self, audio, valid_lens=None, max_sec=None):
        """
        audio: (T,) float32 @16k, 或 (B, T)
        valid_lens: (B,) 每个音频的有效采样点数 (未 pad 的真实长度)
        返回: (B, 384) 全局嵌入 (仅对有效语音帧平均, 避免 pad 静音稀释)
        """
        audio = audio.float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            if valid_lens is not None:
                valid_lens = valid_lens.unsqueeze(0)
        B, T = audio.shape
        if max_sec is not None:
            max_s = int(max_sec * SR)
            if T > max_s:
                audio = audio[:, :max_s]
                if valid_lens is not None:
                    valid_lens = valid_lens.clamp(max=max_s)
        mel = log_mel_spectrogram(audio, n_mels=80)  # (B, 80, T')
        frames = self.encoder(mel)                    # (B, T'', 384), T'' ≈ T'/2 (conv2 stride=2)
        if valid_lens is None:
            return frames.mean(dim=1)
        n_frames_out = frames.shape[1]
        n_valid_mel = (valid_lens // HOP_LENGTH).clamp(1, mel.shape[-1])
        n_valid = (n_valid_mel // 2).clamp(1, n_frames_out).to(audio.device)  # conv2 stride=2
        mask = (torch.arange(n_frames_out, device=audio.device).unsqueeze(0) < n_valid.unsqueeze(1))
        mask = mask.unsqueeze(-1).float()
        return (frames * mask).sum(1) / mask.sum(1).clamp(min=1)

    @torch.no_grad()
    def embed_with_frames(self, audio, valid_lens=None, max_sec=None, n_frames=24):
        """返回 (global(384,), frames(B, n_frames, 384)) — 帧嵌入插值到固定帧数"""
        audio = audio.float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            if valid_lens is not None:
                valid_lens = valid_lens.unsqueeze(0)
        B, T = audio.shape
        if max_sec is not None:
            max_s = int(max_sec * SR)
            if T > max_s:
                audio = audio[:, :max_s]
                if valid_lens is not None:
                    valid_lens = valid_lens.clamp(max=max_s)
        mel = log_mel_spectrogram(audio, n_mels=80)
        frames = self.encoder(mel)                       # (B, T'', 384)
        # 插值到固定帧数
        if frames.shape[1] != n_frames:
            frames = F.interpolate(frames.transpose(1, 2), size=n_frames,
                                   mode='linear', align_corners=False).transpose(1, 2)
        if valid_lens is None:
            return frames.mean(dim=1), frames
        n_frames_out = mel.shape[-1] // 2
        n_valid_mel = (valid_lens // HOP_LENGTH).clamp(1, mel.shape[-1])
        n_valid = (n_valid_mel // 2).clamp(1, max(n_frames_out, 1)).to(audio.device)
        # 按有效比例缩放 (插值后帧数 = n_frames)
        valid_ratio = n_valid.float() / n_frames_out
        n_valid_fixed = (valid_ratio * n_frames).clamp(1, n_frames).long()
        mask = (torch.arange(n_frames, device=audio.device).unsqueeze(0) < n_valid_fixed.unsqueeze(1))
        mask = mask.unsqueeze(-1).float()
        pooled = (frames * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled, frames


if __name__ == '__main__':
    import os
    import soundfile as sf
    from pathlib import Path
    BASE = Path(os.environ.get("MISP_ROOT", Path(__file__).resolve().parent))
    ckpt = BASE / "辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt"
    enc = WhisperEncoder(ckpt)
    wav = BASE / "val/dev_seen/wav/pair_000001_enroll.wav"
    d, sr = sf.read(str(wav), dtype='float32')
    emb = enc.embed(torch.from_numpy(d), max_sec=1.5)
    print("embed shape:", emb.shape)
    print("norm:", emb.norm().item(), "mean:", emb.mean().item(), "std:", emb.std().item())
    assert emb.shape == (1, 384) and torch.isfinite(emb).all()
    print("OK")
