# MISP-QEKS 提升到 0.7+ 的改进 Loop 计划

> 当前最佳: **AUC=0.6466** (V3, 仅 MFCC 音频) | 目标: **0.7+** | 数据: 500K train + 三模态

---

## 现状诊断

| 维度 | 当前状态 | 问题 |
|------|---------|------|
| 数据量 | 只用 val/ (10K) | train/ 有 500K 样本完全没用到 |
| 音频 | MFCC 自训练 CNN | 没用预训练模型 (Wav2Vec2/HuBERT) |
| 文本 | **完全没用** | enrollment_text 直接可用，极其重要 |
| 视频 | **完全没用** | 口唇视频可用于噪声环境 |
| 建模 | BCE Loss | 没用对比学习 / FocalLoss |
| 验证 | dev_unseen 单独 | 缺少 dev_seen 的 AUC 评估 |

---

## 四个 Loop 迭代 (从易到难，逐步叠加)

### Loop 1: 数据规模 + 文本模态 (预期 AUC: 0.68-0.72)

**这是最重要的改进。** 500K 训练数据 + enrollment_text 直接提供关键词信息。

#### 1.1 使用 train/ 目录的完整 500K 数据

```
train/
├── cases/          # JSON 描述每个 case 的数据组成
├── data/
│   ├── meetingroom_data.json   # 会议室场景训练数据索引
│   └── workflow_data.json      # 工作流场景训练数据索引
└── tool_specs.json
```

需要解析 `meetingroom_data.json` 和 `workflow_data.json` 获取 500K 训练样本的文件路径映射。

#### 1.2 引入文本编码器

Enrollment text 就是关键词本身（如 "打开空调"、"播放音乐"），直接用字符级 CNN / Transformer 编码：

```python
class TextEncoder(nn.Module):
    """将 enrollment 关键词文本编码为向量"""
    def __init__(self, vocab_size=5000, emb_dim=256):
        # 字符级 embedding + 1D CNN + attention pooling
        self.char_emb = nn.Embedding(vocab_size, emb_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 256, 3, padding=1), nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1), nn.ReLU(),
        )
        self.attn_pool = nn.Conv1d(256, 1, 1)
        self.proj = nn.Linear(256, emb_dim)

    def forward(self, text_tokens):  # (B, max_len)
        x = self.char_emb(text_tokens).transpose(1, 2)  # (B, 256, L)
        x = self.conv(x)  # (B, 256, L)
        w = self.attn_pool(x).softmax(-1)
        x = (w * x).sum(-1)  # (B, 256)
        return self.proj(x)  # (B, emb_dim)
```

#### 1.3 多模态融合改进

```
Enrollment: [audio_emb (256)] + [text_emb (256)] -> enroll_emb (256)
Query:      [audio_emb (256)]                       -> query_emb  (256)
                                                          |
                                          cross-attention matching
                                                          |
                                          [enroll, query, |diff|, prod] -> classifier
```

Fusion 方式：将 text_emb 与 audio_emb concat 后过 MLP，或直接相加。

#### 1.4 改进的损失函数

```python
# Focal Loss — 关注难分样本
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

# 对比损失 — 拉近正样本对，推远负样本对
def contrastive_loss(enroll_emb, query_emb, labels, temperature=0.07):
    sim = F.cosine_similarity(enroll_emb, query_emb, dim=-1)
    pos = labels * (1 - sim) ** 2
    neg = (1 - labels) * F.relu(sim - 0.5) ** 2  # margin-based
    return (pos + neg).mean()

# 总损失 = BCE/Focal + λ * contrastive
```

#### 1.5 Loop 1 代码结构

```python
# baseline_v4_loop1.py

class KWSpotterV4(nn.Module):
    def __init__(self):
        self.audio_encoder = AudioEncoderV3()     # 复用 V3
        self.text_encoder = TextEncoder()          # 新增
        self.fusion = nn.Linear(512, 256)          # audio+text -> 256
        self.matcher = CrossAttentionMatcher()
        self.classifier = MLP(256*4, 1)
    
    def forward(self, enroll_audio, enroll_text_tokens, query_audio):
        ae = self.audio_encoder(enroll_audio)     # (B, 256)
        te = self.text_encoder(enroll_text_tokens) # (B, 256)
        enroll_emb = self.fusion(torch.cat([ae, te], -1))  # (B, 256)
        query_emb, q_frames = self.audio_encoder(query_audio, return_frames=True)
        # ... cross-attention + classify
```

#### 1.6 Loop 1 超参数搜索

```python
hp_grid_loop1 = [
    {'lr': 1e-3, 'bs': 256, 'epochs': 30, 'hidden': 320, 'text_dim': 256, 'focal_gamma': 2.0, 'tag': 'v4_base'},
    {'lr': 5e-4, 'bs': 256, 'epochs': 40, 'hidden': 384, 'text_dim': 320, 'focal_gamma': 1.5, 'tag': 'v4_wide'},
    {'lr': 3e-4, 'bs': 128, 'epochs': 50, 'hidden': 384, 'text_dim': 384, 'focal_gamma': 2.0, 'tag': 'v4_deep'},
    {'lr': 5e-4, 'bs': 256, 'epochs': 35, 'hidden': 320, 'text_dim': 256, 'focal_gamma': 1.0, 'tag': 'v4_focal1'},
]
```

---

### Loop 2: 预训练模型替换 MFCC (预期 AUC: 0.70-0.75)

MFCC 丢失了大量语音信息。使用预训练模型提取更好的声学表征。

#### 2.1 可选的预训练模型

| 模型 | 参数量 | 下载方式 | 优点 |
|------|-------|---------|------|
| **Wav2Vec2-Base** | 95M | `torchaudio.pipelines.WAV2VEC2_BASE` | 无需额外下载，torchaudio 内置 |
| **HuBERT-Base** | 95M | `torchaudio.pipelines.HUBERT_BASE` | 同上，torchaudio 内置 |
| **WavLM-Base** | 94M | `microsoft/wavlm-base` (huggingface) | 噪声鲁棒性更强 |
| **Whisper-Tiny** | 39M | `openai/whisper-tiny` | 多语言，encoder 输出 384 维 |

#### 2.2 推荐方案：Wav2Vec2-Base (torchaudio 内置，无需下载)

```python
import torchaudio

class Wav2Vec2Encoder(nn.Module):
    def __init__(self, freeze_base=True):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.model = bundle.get_model()
        if freeze_base:
            for p in self.model.parameters():
                p.requires_grad = False
        # Wav2Vec2 输出 768 维，投影到 256
        self.proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
        )
    
    def forward(self, audio, sr=16000):
        """audio: (B, T) raw waveform"""
        features, _ = self.model.extract_features(audio)
        # features: list of layer outputs, 取最后一层
        x = features[-1]  # (B, T', 768)
        x = x.mean(dim=1)  # Global pooling
        return self.proj(x)
```

#### 2.3 使用方式

Wav2Vec2 需要原始波形输入（不是 MFCC）。修改数据加载：

```python
def load_audio_raw(wav_path, max_sec, sr=16000):
    data, orig_sr = sf.read(str(wav_path), dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)
    audio = torch.from_numpy(data).float()
    if orig_sr != 16000:
        audio = torchaudio.functional.resample(audio, orig_sr, 16000)
    max_s = int(max_sec * 16000)
    if len(audio) > max_s:
        audio = audio[:max_s]
    elif len(audio) < max_s:
        audio = F.pad(audio, (0, max_s - len(audio)))
    return audio  # (T,)
```

#### 2.4 双路方案（推荐）

```python
class HybridEncoder(nn.Module):
    """Wav2Vec2 + MFCC 双路，互补信息"""
    def __init__(self):
        self.wav2vec = Wav2Vec2Encoder(freeze_base=True)  # 768 -> 256
        self.mfcc_encoder = AudioEncoderV3()               # 120 -> 256
        self.fusion = nn.Linear(512, 256)
    
    def forward(self, audio_raw, mfcc_feat):
        w2v_emb = self.wav2vec(audio_raw)    # (B, 256)
        mfcc_emb = self.mfcc_encoder(mfcc_feat)  # (B, 256)
        return self.fusion(torch.cat([w2v_emb, mfcc_emb], -1))
```

#### 2.5 Loop 2 搜索方向

```python
hp_grid_loop2 = [
    # Wav2Vec2 + Text
    {'encoder': 'wav2vec2', 'lr': 2e-4, 'bs': 64, 'epochs': 20, 'hidden': 256, 'tag': 'w2v2_ft'},
    # Wav2Vec2 unfrozen
    {'encoder': 'wav2vec2', 'lr': 1e-4, 'bs': 32, 'epochs': 20, 'hidden': 256, 'freeze_base': False, 'tag': 'w2v2_full'},
    # Hybrid: Wav2Vec2 + MFCC
    {'encoder': 'hybrid', 'lr': 2e-4, 'bs': 64, 'epochs': 25, 'hidden': 320, 'tag': 'hybrid'},
    # HuBERT
    {'encoder': 'hubert', 'lr': 2e-4, 'bs': 64, 'epochs': 20, 'hidden': 256, 'tag': 'hubert'},
]
```

---

### Loop 3: 视频模态 + 音素特征 (预期 AUC: 0.73-0.78)

#### 3.1 视频模态

MISP-QEKS 提供口唇视频 (mp4)。可以用：
- **Lip-reading 预训练模型**（如 LipNet, AV-HuBERT）
- **简化方案**：用预训练的 video encoder (ResNet3D / ViViT) 提取帧级特征

```python
# 简化方案: OpenCV 提取口唇帧 + 预训练 CNN
import cv2

def extract_lip_frames(video_path, max_frames=50):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(torch.from_numpy(gray).float() / 255.0)
    cap.release()
    # Pad
    while len(frames) < max_frames:
        frames.append(torch.zeros_like(frames[0]))
    return torch.stack(frames)  # (T, H, W)
```

> 注意：如果视频特征提取太慢，可以跳过此 loop 或仅用于关键场景。

#### 3.2 音素特征 (FA — Forced Alignment)

val/ 目录下有 `fa/` 子目录包含 TextGrid 文件，提供音素时间对齐信息。可用于：
- Hard negative mining：找发音相似的负样本
- 音素序列匹配：enrollment 音素在 query 音素中出现的概率

```python
def parse_textgrid_phones(tg_path):
    """提取音素序列"""
    with open(tg_path) as f:
        content = f.read()
    phones = []
    in_phone_tier = False
    for line in content.split('\n'):
        if 'name = "phones"' in line:
            in_phone_tier = True; continue
        if in_phone_tier:
            m = re.search(r'text = "([^"]*)"', line)
            if m and m.group(1) and m.group(1) != 'sp':
                phones.append(m.group(1))
    return phones

def phone_match_score(enroll_phones, query_phones):
    """enrollment 音素在 query 中的最长匹配比例"""
    n, m = len(enroll_phones), len(query_phones)
    if n == 0:
        return 0.0
    # 最长公共子序列
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            if enroll_phones[i-1] == query_phones[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m] / n
```

#### 3.3 音素 Hard Negative Mining

在训练时，找那些 enrollment 音素和 query 音素部分匹配但 label=0 的样本，将其作为 hard negatives，增加其训练权重。

---

### Loop 4: 集成 + 后处理 (预期 AUC: 0.75-0.80)

#### 4.1 模型集成

```python
# 对多个模型的预测做加权平均
ensemble_weights = {
    'v4_text_audio': 0.35,      # Loop 1 最佳
    'w2v2_text': 0.40,          # Loop 2 最佳
    'hybrid_mfcc_w2v2': 0.15,   # Loop 2 双路
    'video_lip': 0.10,          # Loop 3 (如果有)
}
final_prob = sum(w * model.predict(x) for model, w in zip(models, weights))
```

#### 4.2 测试时增强 (TTA)

```python
def predict_with_tta(model, enroll_audio, query_audio, n_aug=3):
    probs = []
    # 原始预测
    probs.append(model.predict(enroll_audio, query_audio))
    # 加轻微噪声
    for _ in range(n_aug):
        noisy_query = query_audio + torch.randn_like(query_audio) * 0.005
        probs.append(model.predict(enroll_audio, noisy_query))
    return torch.stack(probs).mean()
```

#### 4.3 后处理校准

对预测概率做 Platt Scaling 或 Isotonic Regression 校准，确保 seen 和 unseen 集概率分布一致。

---

## 实施优先级

```
必须做 (缺一不可到 0.7):
  ✅ Loop 1.1: 500K 训练数据        — 数据量是当前 50 倍
  ✅ Loop 1.2: 文本模态            — 关键词文本是强信号
  ✅ Loop 1.4: FocalLoss           — 处理类别不平衡

强烈推荐:
  ✅ Loop 2.1: Wav2Vec2           — 预训练语音表征，torchaudio 内置
  ✅ Loop 1.5: 对比损失            — 增强 embedding 区分度
        
锦上添花:
  ⭐ Loop 3.2: 音素特征           — 增强 hard negative 处理
  ⭐ Loop 4.1: 模型集成           — 多模型互补
  ⭐ Loop 4.2: TTA                — 测试时增强
```

---

## Loop 1 实施步骤 (最小可行路径到 0.7)

### Step 1: 探索 train/ 目录数据结构

```bash
# 查看数据索引文件结构
python -c "import json; d=json.load(open('train/data/meetingroom_data.json')); print(list(d.keys())[:5], len(d))"
```

### Step 2: 写 baseline_v4_loop1.py

基于 baseline_v3.py，添加：
1. `TextEncoder` 类
2. `load_train_data()` — 解析 `meetingroom_data.json` + `workflow_data.json`
3. 更新 `KWSpotterV4` — 加入 text 输入和 fusion
4. `FocalLoss` + contrastive loss
5. 更大的 batch_size (利用更多数据)

### Step 3: 运行 Loop 1

```bash
python baseline_v4_loop1.py
```

### Step 4: 评估

```bash
# 如果没有 dev 上的真实评估，可以分割训练集做交叉验证
python make_submission_v4.py
```

---

## 关键超参数建议

| 参数 | V3 (旧) | V4 Loop1 (建议) | 原因 |
|------|--------|----------------|------|
| batch_size | 64-128 | 256-512 | 数据量 50x |
| lr | 3e-4 | 5e-4 ~ 1e-3 | 更大 batch → 更大 lr |
| hidden | 256-384 | 320-512 | 多模态需要更多容量 |
| epochs | 50-80 | 20-40 | 数据多就不需要太多 epoch |
| dropout | 0.3 | 0.3-0.4 | 防止过拟合 |
| optimizer | AdamW | AdamW + warmup | 稳定训练 |
| scheduler | ReduceLROnPlateau | CosineAnnealing | 更好收敛 |

---

## 常见陷阱

1. **OOM**: 500K 样本不能全部加载到内存。用 **惰性加载** — 只缓存文件路径，每次 `__getitem__` 时读取和提取特征。
2. **I/O 瓶颈**: 使用多进程 DataLoader (`num_workers=4`) 并行读取音频文件。
3. **过拟合 dev**: 用 train 的 90% 做训练、10% 做本地验证，dev 仅做最终评估。
4. **类别不平衡**: train 中正负样本比例约 1:1？需确认。若不平衡，用 FocalLoss 或过采样。
5. **音频长度**: enrollment 通常 1-2 秒，query 通常 3-4 秒。截断过长音频即可。

---

## 期望提分路线图

```
V3 (当前):    0.6466  (MFCC + 10K 数据)
    ↓ + 500K 数据
V4a:          ~0.66   (更多数据 → 更好泛化)
    ↓ + 文本模态
V4b:          ~0.68   (关键词文字 = 强信号)
    ↓ + FocalLoss + 对比损失
V4c:          ~0.70   (更好的损失函数)
    ↓ + Wav2Vec2
V5:           ~0.73   (预训练语音表征)
    ↓ + 集成 + TTA  
V6:           ~0.75+  (多模型融合)
```
