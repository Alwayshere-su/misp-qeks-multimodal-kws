# MISP-QEKS 优化进度记忆 (保存于 2026-08-05 ~20:55)

> **历史文件**：这里记录的是当时的 0.82755 状态。后续多模板路线已达到 0.90042，最新结论见根 `README.md`；本文件仅保留当时的断点与诊断。
> 文中的 `output_v3`~`output_v6` 已于 2026-10-20 删除，`output_v5` 里的旧权重属于 baseline_v5，与根目录 `train_matcher_v5.py` 无关。
>
> 用户下次说【继续多模态】即可从本文件续接。

## 目标与现状
- 当前线上分数: **0.82755** (来自 output_p4 等基于 10K val 训练的模型提交)
- 目标: **0.96933+** (0.5*AUC_seen + 0.5*AUC_blind)
- 提交限制: 每天最多 3 次

## 环境要点 (重要!)
- Windows + PowerShell; 工作目录 `D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛`
- GPU: RTX 5060 8GB (WDDM); torch 2.11.0+cu128, CUDA 可用
- **沙箱内 import torch/sklearn/pandas 会报 ctypes DLL 拒绝访问** → 所有 ML 命令需 `require_escalated`
- 已批准前缀规则: `["python","-c"]` (可运行任意 python -c 代码), `["python","scripts/_search_p16.py"]`, `["python","scripts/_verify_dois.py"]`
- 磁盘空间无法查询 (fsutil/CIM 被拒); 现有大文件: train.zip 20GB, test.zip 4.3GB, output_v3/*.pkl 各 14.5GB
- 网络受限 (未测试 huggingface 下载; 本地只有 whisper-tiny + lipreading LRW 预训练模型)

## 关键诊断结论 (决定后续方向!)
1. **val 集是"文本匹配"任务**: label=1 ⟺ enrollment_txt 出现在 query_txt 中 (val 100%, train 99.95% 一致)。
   → p4 模型在 val 上 AUC=1.0000 (纯文本规则即可 1.0)。**val 是"简单代理"，线上测试难得多**。
2. **Whisper-tiny ASR 完全不可用**: 直接解码输出重复符号/幻觉文本; 温度回退+重复抑制后仍完全乱码。
   fuzzy-match AUC≈0.5, 精确命中 0/400。→ **放弃 ASR 文本匹配路线**。
3. **音频余弦基线 (whisper-tiny 全局嵌入)**:
   - dev_seen: pos率20%, 音频cos AUC=0.8125, 视频cos AUC=0.5721 (seen 难)
   - dev_unseen: 音频cos AUC=0.9542, 视频cos AUC=0.9065 (unseen 易)
   - 帧级 mean-of-max: seen 0.8237, unseen 0.9503
4. **500K train 从未被用于训练 matcher!** `output_whisper/train_feats.npz` 不存在。
   现有模型 (p1-p4) 都只在 10K val 上训练 → 线上泛化差是必然。**这是最大机会**。
5. 训练标签规则同样为文本包含 (抽查 2000: 99.95%), pos率≈19.75%。

## 现有资产
- `output_whisper/val_feats.pkl`: val 10K, 含 enroll_emb(384)/enroll_frames(24,384 fp16)/query_emb/query_frames/label/enroll_txt/query_txt
- `output_whisper/test_inset.pkl`, `test_outset.pkl`: 测试各 50K pairs (含 frames)
- `output_lip/val_lip.pkl`, `test_lip_inset.pkl`, `test_lip_outset.pkl`: 视频嵌入 256 维
- `辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt`, `lipreading/lipreading_LRW_0.8018.pt`
- `辅助资源/snr_map_train_dev/*.snr_map`: 各样本 SNR 映射 (可做加权/数据选择)
- 模型权重: output_p1..p4 (best_full.pt 等), output_v5 (best_v5_*.pt)
- 提交样例在 submission/; 各 make_submission_p*.py 生成 CSV

## 已写新脚本 (opt/ 目录)
- `opt/diag_features.py` — 特征余弦AUC诊断 (运行成功)
- `opt/diag_val2.py` — p4模型+文本+帧在val的AUC (运行成功: p4=1.0, 文本=1.0, 帧 seen 0.8237/unseen 0.9503)
- `opt/check_asr_output.py`, `opt/asr_proper.py` — ASR质量检查 (结论: 不可用)
- `opt/check_train_labels.py` — train标签规则验证 (99.95% 文本包含)
- `opt/extract_train_feats_v2.py` — **500K train whisper全局嵌入提取 (分块+checkpoint)** → output_whisper/train_global/
- `opt/train_seqmatcher.py` — **帧级注意力匹配模型** (SeqMatcher: 帧投影→归一化→sim矩阵→attention→per-frame match + 全局相似度 + 文本), 支持 cv / s2u 模式

## 执行状态 (中断点)
- 特征提取 **未成功启动** (第一次 Start-Process 参数被引号搞坏 SyntaxError; 第二次被中断)。
  正确启动方式: `Start-Process python -ArgumentList @('-u','opt/extract_train_feats_v2.py') -WindowStyle Hidden -RedirectStandardOutput opt\extract_train_v2.log -RedirectStandardError opt\extract_train_v2.err -WorkingDirectory <BASE>`
  - `output_whisper/train_global/` 目录**尚未创建**, 日志为空
- 观察到 python 进程: PID 40808 (20:50:59 启动, CPU高, 可能旧任务/干扰), 52364/55064/55128 (可能是失败的提取尝试残留) — 恢复时先确认哪些需要清理

## 下一步计划 (按优先级)
1. **启动并完成 500K train 特征提取** (预计 20-40 min, 全局嵌入即可, 约1.5GB)。可选加帧 (18GB, 需确认磁盘)。
2. **在 10K val 上快速验证 SeqMatcher 架构** (`opt/train_seqmatcher.py --mode both`):
   - 对比 p4 (cv5=0.987, s2u≈0.797) — 若帧级匹配明显更好则上 500K
3. **在 500K train 上训练 SeqMatcher** (或先训全局MLP MatcherV5式 baseline 作为快速提分提交):
   - 验证集用 val (seen/unseen 分开报), 也留 train 内 holdout 检查过拟合
4. **生成提交**: 用测试特征 (test_inset/outset.pkl + frames) 预测, 与现有 p1/p3/p4 预测做加权集成
5. (可选) 微调 whisper-tiny encoder 做任务适应; (可选) 用 SNR map 做难样本加权/增强
6. (可选) 用 LipReading 视频帧特征做帧级视频匹配 (视频全局 cos 在 seen 只有 0.57, 收益不确定)

## 提交格式提醒
CSV 表头 `id,type,prediction`; inset/outset 各 50000 行; id 从 0 开始 (pair_000001 对应 id=0);
线上分 = 0.5*AUC_inset + 0.5*AUC_outset; 每天限 3 次提交。

## 经验教训
- val 集被文本规则"污染" (label 由文本包含决定), 本地 CV 高不代表线上好 → 训练集要用 500K, 别只靠 val
- whisper-tiny ASR 对噪声会议音频幻觉严重, 别浪费时间
- 测试集没有 query 文本, 只能靠音频/视频匹配 + enrollment 文本条件
