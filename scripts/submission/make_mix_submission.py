"""混合提交: inset 用 500k train 模型, outset 用 p4 foldavg (词表覆盖互补)"""
import csv
from pathlib import Path

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def load(path):
    inset, outset = {}, {}
    for r in csv.DictReader(open(path, encoding='utf-8')):
        (inset if r['type'] == 'inset' else outset)[int(r['id'])] = float(r['prediction'])
    return inset, outset


def main():
    train_model = load(BASE / 'submission/submit_500k.csv')          # inset 用 (train 词表内)
    p4_outset = load(BASE / 'submission/submit_p4_foldavg_test.csv') # outset 用 (p4 见过 test_unseen 词)
    out_path = BASE / 'submission/submit_mix_train_seen_p4_unseen.csv'
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            f.write(f"{i},inset,{train_model[0].get(i, 0.5):.6f}\n")
        for i in range(50000):
            f.write(f"{i},outset,{p4_outset[1].get(i, 0.5):.6f}\n")
    print(f'-> {out_path}', flush=True)
    # 另一个版本: outset 用 p1p4_ensemble5 (线上 0.82755 的来源)
    p1p4 = load(BASE / 'submission/submit_p1p4_ensemble5.csv')
    out_path2 = BASE / 'submission/submit_mix_train_seen_p1p4_unseen.csv'
    with open(out_path2, 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            f.write(f"{i},inset,{train_model[0].get(i, 0.5):.6f}\n")
        for i in range(50000):
            f.write(f"{i},outset,{p1p4[1].get(i, 0.5):.6f}\n")
    print(f'-> {out_path2}', flush=True)


if __name__ == '__main__':
    main()
