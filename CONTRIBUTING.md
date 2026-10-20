# 贡献指南

这是一个比赛归档仓库，主要目的是记录并复现 2026 年 MISP-QEKS 挑战赛的最终方案，而不是持续维护的库。仍然欢迎 issue 和 PR，但请先看下面的边界。

## 可以做什么

- 报告文档错误、失效链接、无法运行的脚本。
- 指出复现步骤中的缺漏，或提交更小的、可验证的改进。
- 讨论方法本身（例如为什么无监督多模板比监督模型泛化更好）。

## 请不要提交

- 官方数据集、特征缓存、模型权重或任何 `output_*` 目录下的文件（`.gitignore` 已排除）。
- 新的实验提交 CSV；`submission/` 只保留 5 个有分数记录的版本。
- 与本项目无关的格式化改动或大规模重写。

## 本地检查

提交 PR 前请确认：

```bash
# 1) 仓库内所有 Python 文件可编译
git ls-files '*.py' -z | xargs -0 python -m py_compile

# 2) 提交文件格式与最优文件哈希
python tests/check_submissions.py
```

第 2 项不需要安装任何第三方依赖，CI 也只跑这两步。

## 代码风格

- Python 3.12，4 空格缩进，遵循 PEP 8；改动尽量贴近文件原有写法。
- 路径不要写死本机绝对路径，统一用 `MISP_ROOT` 或 `Path(__file__)` 推导。
- 不在仓库里留下临时脚本、日志和缓存。

## 提交信息

用一句话说清改了什么，例如：

```text
docs: fix invalid link in submission README
fix: make make_submission_p4.py importable from repo root
```
