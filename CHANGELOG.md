# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 的组织方式，版本号使用 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.0.0] - 2026-10-20

首个作品集版本：整理比赛代码、文档与最终提交。

### Added

- 作品集 README：赛事成绩、分数演进、Mermaid 链路图、复现步骤、踩坑结论。
- `docs/`：赛题概要、资产与目录说明、旧特征说明、整理记录，历史笔记移入 `docs/history/`。
- `submission/`：保留 5 个有关键分数的提交 CSV，并附 SHA256 与生成命令。
- `tests/check_submissions.py` 与 GitHub Actions CI：校验提交格式与最优文件哈希。
- `CITATION.cff`、`CONTRIBUTING.md`、`CODE_OF_CONDUCT.md`、`.editorconfig`。
- `requirements.txt` 固定为实测版本（Python 3.12 / torch 2.11 / numpy 1.26）。

### Changed

- 脚本项目根改为 `MISP_ROOT` 环境变量或按 `__file__` 推导，不再写死本机绝对路径。
- 修复 `scripts/submission/make_submission_p4.py` 的 `sys.path` 少一级问题。
- 仓库只保留代码、文档和 5 个提交文件；其余 81 个实验提交移到仓库外归档。

### Removed

- 删除 40.2 GB 已废弃特征与 23 个旧基线权重（`output_v3`–`output_v6`），见 `docs/旧特征40GB说明.md`。
- 清空提交历史并压缩为单个根提交，`.git` 从 53 MB 降到 3.3 MB。
