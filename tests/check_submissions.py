#!/usr/bin/env python3
"""校验仓库内保留的 5 个提交文件。

不依赖第三方库，CI 与本地都能直接跑：

    python tests/check_submissions.py

检查项：
1. `submission/` 下恰好是约定的 5 个 CSV，没有缺失或多余的实验版本；
2. 每个文件表头为 `id,type,prediction`，共 100,000 行；
3. inset / outset 各 50,000 行，id 覆盖 0–49999 且无重复；
4. prediction 是 [0, 1] 内的有限浮点数；
5. 最终最优提交 `submit_gf_dtw.csv` 的 SHA256 与归档值一致。
"""

from __future__ import annotations

import csv
import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "submission"

EXPECTED_FILES = {
    "submit_p4_full.csv",       # 0.82755
    "submit_mt_split.csv",      # 0.89143
    "submit_mt_gf_split.csv",   # 0.89995
    "submit_final_boost.csv",   # 0.89386
    "submit_gf_dtw.csv",        # 0.90042（最终最优）
}

# 归档时的字节级指纹；重新生成若改变数值应同步更新这里
GOLDEN_SHA256 = {
    "submit_gf_dtw.csv": "b77b5ffe90ea23653781e562582ffff02b9504697e32aded7c34c5397e8142a7",
}

HEADER = ["id", "type", "prediction"]
ROWS_PER_TYPE = 50_000
ID_RANGE = range(ROWS_PER_TYPE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    seen: dict[str, set[int]] = {"inset": set(), "outset": set()}
    n_rows = 0

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return [f"{path.name}: 空文件"]

        if header != HEADER:
            return [f"{path.name}: 表头是 {header!r}，期望 {HEADER!r}"]

        for lineno, row in enumerate(reader, start=2):
            n_rows += 1
            if len(row) != 3:
                errors.append(f"{path.name}:{lineno}: 列数 {len(row)}，期望 3")
                continue

            raw_id, kind, raw_pred = row
            if kind not in seen:
                errors.append(f"{path.name}:{lineno}: 未知 type {kind!r}")
                continue
            try:
                sample_id = int(raw_id)
            except ValueError:
                errors.append(f"{path.name}:{lineno}: id 不是整数 {raw_id!r}")
                continue
            if sample_id not in ID_RANGE:
                errors.append(f"{path.name}:{lineno}: id {sample_id} 超出 0–49999")
            elif sample_id in seen[kind]:
                errors.append(f"{path.name}:{lineno}: {kind} 重复 id {sample_id}")
            else:
                seen[kind].add(sample_id)

            try:
                pred = float(raw_pred)
            except ValueError:
                errors.append(f"{path.name}:{lineno}: prediction 不是数字 {raw_pred!r}")
                continue
            if not math.isfinite(pred) or not 0.0 <= pred <= 1.0:
                errors.append(f"{path.name}:{lineno}: prediction {pred} 不在 [0, 1]")

    if n_rows != 2 * ROWS_PER_TYPE:
        errors.append(f"{path.name}: 共 {n_rows} 行，期望 {2 * ROWS_PER_TYPE}")
    for kind, ids in seen.items():
        if len(ids) != ROWS_PER_TYPE:
            missing = len(ID_RANGE) - len(ids)
            errors.append(f"{path.name}: {kind} 只有 {len(ids)} 个唯一 id（缺 {missing}）")

    return errors


def main() -> int:
    if not SUBMISSION_DIR.is_dir():
        print(f"FAIL: 找不到目录 {SUBMISSION_DIR}")
        return 1

    actual = {p.name for p in SUBMISSION_DIR.glob("*.csv")}
    errors: list[str] = []
    for name in sorted(EXPECTED_FILES - actual):
        errors.append(f"缺少提交文件 submission/{name}")
    for name in sorted(actual - EXPECTED_FILES):
        errors.append(f"出现计划外的提交文件 submission/{name}")

    for name in sorted(EXPECTED_FILES & actual):
        path = SUBMISSION_DIR / name
        errors.extend(check_file(path))
        expected_hash = GOLDEN_SHA256.get(name)
        if expected_hash:
            actual_hash = sha256(path)
            if actual_hash != expected_hash:
                errors.append(
                    f"{name}: SHA256 {actual_hash} != 归档值 {expected_hash}"
                )
        print(f"OK   {name}")

    if errors:
        print("\n校验失败：", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"\n全部通过：{len(EXPECTED_FILES & actual)} 个提交文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
