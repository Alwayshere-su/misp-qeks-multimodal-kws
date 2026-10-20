"""p4 声学模型 5 折 OOF 打分 (无泄漏) — 复现 StratifiedKFold(5, shuffle, seed=42)"""
import pickle
import sys
from pathlib import Path

import torch
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_whisper_p4 import load_data, evaluate, MatcherV4, DEVICE, SEED

BASE = Path(r"D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")


def main():
    items = load_data()
    y_all = [it['label'] for it in items]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = [None] * len(items)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(items, y_all)):
        model = MatcherV4().to(DEVICE)
        model.load_state_dict(torch.load(BASE / f'output_p4/best_fold{fold}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        auc, preds = evaluate(model, [items[i] for i in va_idx], DEVICE)
        print(f'fold{fold}: OOF AUC={auc:.6f} n_va={len(va_idx)}', flush=True)
        for i, p in zip(va_idx, preds):
            oof[i] = p
    # 分 split 汇总
    for split in ['dev_seen', 'dev_unseen']:
        idx = [i for i, it in enumerate(items) if it['split'] == split]
        from sklearn.metrics import roc_auc_score
        y = [items[i]['label'] for i in idx]
        p = [oof[i] for i in idx]
        print(f'[{split}] OOF AUC={roc_auc_score(y, p):.6f} n={len(y)} pos={sum(y)}', flush=True)
    # 保存 OOF 分数: {split: {pid: score}}
    out = {'dev_seen': {}, 'dev_unseen': {}}
    for i, it in enumerate(items):
        out[it['split']][it['pid']] = oof[i]
    with open(BASE / 'output_p4/oof_scores.pkl', 'wb') as f:
        pickle.dump(out, f)
    print('saved output_p4/oof_scores.pkl')


if __name__ == '__main__':
    main()
