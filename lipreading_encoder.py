"""
LipReading 视频编码器 — 复用 V6 的预训练方案
============================================
输入: 灰度口唇帧 (1, T, 96, 96) -> 输出 256 维嵌入
预训练: 辅助资源/pretrained_feature_models/lipreading/lipreading_LRW_0.8018.pt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class BasicBlock2D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResNet2D(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return x


class LipReadingEncoder(nn.Module):
    """3D frontend + 2D ResNet18 + temporal pool -> 256-dim"""
    def __init__(self, ckpt_path):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.resnet = ResNet2D(BasicBlock2D, [2, 2, 2, 2])
        self.fc = nn.Linear(512, 256)
        self.bnfc = nn.BatchNorm1d(256)

        ckpt = Path(ckpt_path)
        if ckpt.exists():
            sd = torch.load(ckpt, map_location='cpu', weights_only=False)
            our_sd = {}
            for k, v in sd.items():
                if k.startswith('video_frontend.video_frontend.'):
                    our_sd[k.replace('video_frontend.video_frontend.', 'frontend.')] = v
                elif k.startswith('video_frontend.resnet.'):
                    new_k = k.replace('video_frontend.resnet.', '')
                    if new_k.startswith('fc.') or new_k.startswith('bnfc.'):
                        our_sd[new_k] = v
                    else:
                        our_sd['resnet.' + new_k] = v
            missing, unexpected = self.load_state_dict(our_sd, strict=False)
            print(f"  LipReading loaded: {len(our_sd)-len(missing)}/{len(sd)} keys matched"
                  + (f" ({len(missing)} missing)" if missing else ""))

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def embed(self, x):
        """x: (B, 1, T, H, W) 灰度帧 -> (B, 256)"""
        B = x.shape[0]
        x = self.frontend(x)                       # (B, 64, T, H', W')
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.resnet(x)                          # (B*T, 512, H'', W'')
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(B, T, 512)
        x = x.mean(dim=1)                           # (B, 512)
        x = self.bnfc(self.fc(x))                   # (B, 256)
        return x


def extract_video_frames(video_path, max_frames=24):
    """读取 mp4 -> 灰度帧, 线性插值到 max_frames 帧, 返回 (1, 1, T, 96, 96)"""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 96x96
        frames.append(torch.from_numpy(gray).float() / 255.0)
    cap.release()
    if not frames:
        return torch.zeros(1, 1, max_frames, 96, 96)
    t = torch.stack(frames)  # (T, H, W)
    if t.shape[0] != max_frames:
        t = t.unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(max_frames, 96, 96), mode='trilinear', align_corners=False)
        t = t.squeeze(0)
    else:
        t = t.unsqueeze(0)
    return t  # (1, T, 96, 96)


if __name__ == '__main__':
    import time
    BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
    ckpt = BASE / "辅助资源/pretrained_feature_models/lipreading/lipreading_LRW_0.8018.pt"
    enc = LipReadingEncoder(ckpt)
    mp4 = BASE / "val/dev_seen/mp4/pair_000001_enroll.mp4"
    t0 = time.time()
    frames = extract_video_frames(mp4, max_frames=24)
    emb = enc.embed(frames.unsqueeze(0))
    print(f"frames: {frames.shape}, emb: {emb.shape}, time: {time.time()-t0:.2f}s")
    print("OK")
