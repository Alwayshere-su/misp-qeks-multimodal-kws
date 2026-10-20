# 提交文件说明

本目录只保留 5 个有线上分数记录的提交，覆盖从基线到最终最优的完整路径。其余 81 个实验版本已移到仓库外的 `_archive_misp_submissions/`（未删除，只是不进入版本控制），以免把 170 MB 生成物放进 Git。

## 格式

- 表头：`id,type,prediction`
- 共 100,000 行数据：`inset` 50,000 行 + `outset` 50,000 行
- 每种 type 的 `id` 从 0 到 49999，唯一且完整
- `prediction` 为 [0,1] 的概率
- 线上分数：`0.5 × AUC_inset + 0.5 × AUC_outset`

## 保留的提交与分数

| 提交 | 分数 | 说明 |
|---|---:|---|
| `submit_p4_full.csv` | 0.82755 | p4 基线（`output_p4/best_full.pt`） |
| `submit_mt_split.csv` | 0.89143 | 多模板 + p4 |
| `submit_mt_gf_split.csv` | 0.89995 | 全局 + 帧级多模板 + p4 |
| `submit_final_boost.csv` | 0.89386 | 五信号融合 |
| **`submit_gf_dtw.csv`** | **0.90042** | **最终最优**：gf_split + outset 全模板 DTW 起止加权 |

分数来源：p4_full / mt_split / final_boost 见根 README；mt_gf_split 见 `opt/make_iso_submit.py`；gf_dtw 见 `opt/make_gf_dtw_v2.py`。

`submit_gf_dtw.csv` 的 SHA256：

```
b77b5ffe90ea23653781e562582ffff02b9504697e32aded7c34c5397e8142a7
```

## 生成方式

```bash
# p4 基线
python scripts/submission/make_submission_p4.py \
    --model output_p4/best_full.pt \
    --out submission/submit_p4_full.csv

# 多模板（产出 submit_multitemplate.csv 与 submit_mt_full_w*.csv）
python opt/make_multitemplate_submit.py

# 最优提交（基于已归档的 submit_mt_gf_split.csv 加 DTW）
python opt/make_dtw_submit.py
```

脚本默认把项目根解析为仓库根目录，也可以用环境变量 `MISP_ROOT` 覆盖。运行前确认 `output_whisper/`、`output_p4/` 存在。

> `submit_mt_gf_split.csv` 的生成脚本没有保留在仓库中，现有脚本只读取它；因此复现 `submit_gf_dtw.csv` 依赖本目录里归档的该 CSV。

## 已归档的实验提交

以下类型已移到 `_archive_misp_submissions/`，仅作本地留存：

- 无分数记录的后续变体：`gf_dtw_v2~v4`、`iso_500k`、`iso_seq`、`fuse_v2/v3`、`robust_mt*`、`base_mt_v1~v3` 等；
- 6 个恒为 0.5 的退化输出：`frame_only`、`mt_norm_only`、`mt_quality_only`、`mt_robust_only`、`mt_softmax_only`、`phonetic_only`；
- 3 组内容完全相同的重复文件（如 `submit_mt_norm_split.csv` 与 `..._b.csv`）。
