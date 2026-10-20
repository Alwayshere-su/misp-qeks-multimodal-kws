# opt 实验脚本

`opt/` 是比赛期间的实验场：临时脚本、诊断脚本和一次性提交生成脚本都堆在这里。它不是干净的入口；核心模块仍在项目根目录，整理后的脚本在 `scripts/`。

## 当前最优链路

- `make_dtw_submit.py` — 在 `submit_mt_gf_split.csv`（0.89995）基础上给 outset 加全模板 DTW 起止权重，生成 `submission/submit_gf_dtw.csv`（**0.90042，最终最优**）。
- `make_multitemplate_submit.py` — 无监督多模板 + p4 融合，产出 `submit_multitemplate.csv` 与 `submit_mt_full_w0.3/0.5/0.7.csv`；**不生成** `submit_mt_gf_split.csv`（该文件的生成脚本已不在仓库）。
- `frame_dtw.py`、`make_gf_dtw_v2.py` — DTW 实现与权重微调；`make_gf_dtw_v2.py` 生成 `submit_gf_dtw_v2~v4.csv`，这三个没有分数记录。
- `make_fuse_submit.py`、`make_iso_submit.py` — 融合/归因实验：`submit_fuse_v1.csv` 记录 0.8934（掉分），`v2/v3` 与 `iso_*` 无记录。

## 特征与训练

| 脚本 | 作用 |
|---|---|
| `extract_train_feats_v2.py` | 500K train whisper 全局嵌入，输出到 `output_whisper/train_global/`（分块 + checkpoint） |
| `extract_base_test.py` | whisper-base 测试集嵌入 |
| `train_500k_matcher.py` | 500K train 匹配模型，输出 `out_500k/best_500k.pt` |
| `train_seqmatcher.py` | 帧级注意力匹配模型 |
| `retrain_mt.py`、`retrain_mt2.py` | 多模板重训练实验，输出 `out_retrain/` |
| `robust_mt_pipeline.py`、`robust_pipeline.py` | 鲁棒多模板流水线 |

## 诊断与检查

`check_*.py`、`diag_*.py`、`analyze_*.py`、`_check_*.py`、`_eval_*.py` 是当时的诊断脚本，用于检查标签规则、特征偏移、OOV 瓶颈、DTW 融合等。多数只对当时的缓存有效，不要当成通用工具。

## 输出目录

| 路径 | 内容 |
|---|---|
| `out_500k/` | `best_500k.pt`、`best_500k_probe.pt`、500K 提交中间文件 |
| `out_retrain/` | `best_mt.pt`、`best_mt2.pt` |
| `out_seq/` | 帧级模型的 5 折 + s2u 权重、`results_seq.json` |

## 注意

- `PROGRESS.md` 是 **2026-08-05** 的历史断点（当时 0.82755），不是最新状态；最新已知结果看根 `README.md` 和 `docs/旧特征40GB说明.md`。
- `test2.py`、`test3.py`、`test_write.py`、`_gen.py`、`test.txt` 等是临时文件；其中 `test3.py` 和 `_gen.py` 本身有语法错误，不参与任何管线。
- 提交文件统一在 `submission/`，说明见 `submission/README.md`。
