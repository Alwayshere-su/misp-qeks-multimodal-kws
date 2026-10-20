"""多模板匹配提交: query 与同词所有 enroll 音频的最大相似度 (无监督)
用法: python opt/make_multitemplate_submit.py
输出: submission/submit_multitemplate.csv (+ 与 p4_full 融合版)
"""
import csv
import os
import pickle
import re
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get('MISP_ROOT', Path(__file__).resolve().parents[1]))


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def multitemplate_scores(data):
    """返回 {pid: score}, 每词模板用全部 enroll 音频"""
    from collections import defaultdict
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))
    scores = {}
    for w, items in word_items.items():
        templates = np.stack([it['enroll_emb'] for _, it in items])
        tn = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-9)
        for pid, it in items:
            q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
            sims = q @ tn.T
            scores[pid] = float(sims.max())
    return scores


def load_submission(n):
    ins, out = {}, {}
    for r in csv.DictReader(open(BASE / 'submission' / n, encoding='utf-8')):
        (ins if r['type'] == 'inset' else out)[int(r['id'])] = float(r['prediction'])
    return ins, out


def rank_norm(preds):
    from scipy.stats import rankdata
    ids = sorted(preds.keys())
    v = np.array([preds[i] for i in ids])
    r = rankdata(v) / len(v)
    return dict(zip(ids, r))


def main():
    print('加载 test 特征...', flush=True)
    tin = pickle.load(open(BASE / 'output_whisper/test_inset.pkl', 'rb'))
    tout = pickle.load(open(BASE / 'output_whisper/test_outset.pkl', 'rb'))
    print('计算多模板分数...', flush=True)
    si = multitemplate_scores(tin)
    so = multitemplate_scores(tout)
    print(f'inset {len(si)} outset {len(so)}', flush=True)

    # 提交 1: 纯多模板 (归一化到 [0,1] 便于查看; AUC 只依赖排序, 用 rank 保证范围)
    r_i = rank_norm(si)
    r_o = rank_norm(so)
    with open(BASE / 'submission/submit_multitemplate.csv', 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},inset,{r_i.get(pid, 0.5):.6f}\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},outset,{r_o.get(pid, 0.5):.6f}\n")
    print('saved submit_multitemplate.csv', flush=True)

    # 提交 2: 多模板 + p4_full 融合
    full = load_submission('submit_p4_full.csv')
    rf_i, rf_o = rank_norm(full[0]), rank_norm(full[1])
    for w_mt in [0.3, 0.5, 0.7]:
        with open(BASE / f'submission/submit_mt_full_w{w_mt:.1f}.csv', 'w', encoding='utf-8', newline='') as f:
            f.write("id,type,prediction\n")
            for i in range(50000):
                pid = f"pair_{i+1:06d}"
                sc = w_mt * r_i.get(pid, 0.5) + (1 - w_mt) * rf_i.get(i, 0.5)
                f.write(f"{i},inset,{sc:.6f}\n")
            for i in range(50000):
                pid = f"pair_{i+1:06d}"
                sc = w_mt * r_o.get(pid, 0.5) + (1 - w_mt) * rf_o.get(i, 0.5)
                f.write(f"{i},outset,{sc:.6f}\n")
        print(f'saved submit_mt_full_w{w_mt:.1f}.csv', flush=True)


if __name__ == '__main__':
    main()
