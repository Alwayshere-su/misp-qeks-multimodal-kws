"""Verify train label rule: label=1 <=> enrollment text in query text (sample)."""
from pathlib import Path
import random

BASE = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
LABEL = BASE / "train" / "label"

random.seed(0)
pids = [f"pair_{i:06d}" for i in range(1, 500001)]
random.shuffle(pids)

consistent = 0
total = 0
pos_total = 0
pos_consistent = 0
for pid in pids[:20000]:
    parts = open(LABEL / f"{pid}.txt", encoding="utf-8").read().strip().split("\t")
    et = parts[0].replace("enrollment:", "").lower().strip()
    qt = parts[1].replace("query:", "").lower().strip()
    label = int(parts[2].replace("label:", ""))
    rule = 1 if (et and qt and et in qt) else 0
    total += 1
    if rule == label:
        consistent += 1
    if label == 1:
        pos_total += 1
        if rule == 1:
            pos_consistent += 1
print(f"rule consistency: {consistent}/{total} = {consistent/total:.4f}")
print(f"pos rate: {pos_total}/{total} = {pos_total/total:.4f}")
print(f"pos rule consistency: {pos_consistent}/{pos_total} = {pos_consistent/pos_total:.4f}")

# check some samples
for pid in pids[:5]:
    parts = open(LABEL / f"{pid}.txt", encoding="utf-8").read().strip().split("\t")
    print(parts)
