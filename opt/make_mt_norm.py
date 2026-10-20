"""简化对比归一化: max - sim(query, 全局均值向量) + 多层融合提交 (test)
inset: 对比归一化主导 (val seen 验证); outset: max 主导 (val unseen 验证)
"""
import csv
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def compute(data, emb_key='enroll_emb', qkey='query_emb'):
    """返回 max 和 对比归一化 分数"""
    from collections import defaultdict
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))
    all_emb = np.stack([it[emb_key] for items in word_items.values() for _, it in items])
    all_mean = all_emb.mean(0)
    all_mean_n = all_mean / (np.linalg.norm(all_mean) + 1e-9)
    s_max, s_norm = {}, {}
    for w, items in word_items.items():
        templates = np.stack([it[emb_key] for _, it in items])
        tn = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-9)
        for pid, it in items:
            q = it[qkey] / (np.linalg.norm(it[qkey]) + 1e-9)
            sims = q @ tn.T
            s_max[pid] = float(sims.max())
            s_norm[pid] = float(sims.max() - (q @ all_mean_n))
    return s_max, s_norm


def rn(p):
    ids = sorted(p.keys())
    v = np.array([p[i] for i in ids])
    r = rankdata(v) / len(v)
    return dict(zip(ids, r))


def main():
    # val 验证简化版
    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))
    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        sm, sn = compute(data)
        y = [data[p]['label'] for p in sorted(data)]
        a = np.array([sm[p] for p in sorted(data)])
        b = np.array([sn[p] for p in sorted(data)])
        ar, br = rn(sm), rn(sn)
        print(f'{split}: max AUC={roc_auc_score(y, a):.4f} '
              f'norm AUC={roc_auc_score(y, b):.4f} ', flush=True)
        best = 0
        for w in np.arange(0, 1.01, 0.1):
            auc = roc_auc_score(y, w * np.array([ar[p] for p in sorted(data)])
                                + (1 - w) * np.array([br[p] for p in sorted(data)]))
            if auc > best:
                best, bw = auc, w
        print(f'  融合 best w_max={bw:.1f} AUC={best:.4f}', flush=True)

    # test 提交: split 感知
    print('计算 test 分数...', flush=True)
    tin = pickle.load(open(BASE / 'output_whisper/test_inset.pkl', 'rb'))
    tout = pickle.load(open(BASE / 'output_whisper/test_outset.pkl', 'rb'))
    si_m, si_n = compute(tin)
    so_m, so_n = compute(tout)
    ri_m, ri_n = rn(si_m), rn(si_n)
    ro_m, ro_n = rn(so_m), rn(so_n)
    full = {}
    for r in csv.DictReader(open(BASE / 'submission/submit_p4_full.csv', encoding='utf-8')):
        (full.setdefault('inset', {}).setdefault(int(r['id']), float(r['prediction']))
         if r['type'] == 'inset' else full.setdefault('outset', {}).setdefault(int(r['id']), float(r['prediction'])))
    rf_i, rf_o = rn(full['inset']), rn(full['outset'])

    # 内层(多模板): inset w_max=0.2 (norm 主导), outset w_max=0.8
    # 外层(p4): inset 0.5, outset 0.2 (沿用 mt_split)
    def inner(wmax, rmax, rnorm):
        return {i: wmax * rmax.get(i, 0.5) + (1 - wmax) * rnorm.get(i, 0.5) for i in range(50000)}
    mt_i = inner(0.2, ri_m, ri_n)
    mt_o = inner(0.8, ro_m, ro_n)
    with open(BASE / 'submission/submit_mt_norm_split.csv', 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            f.write(f"{i},inset,{0.5 * mt_i[i] + 0.5 * rf_i[i]:.6f}\n")
        for i in range(50000):
            f.write(f"{i},outset,{0.8 * mt_o[i] + 0.2 * rf_o[i]:.6f}\n")
    print('saved submit_mt_norm_split.csv', flush=True)
    # 变体: 内层 inset 用 w_max=0.4
    mt_i2 = inner(0.4, ri_m, ri_n)
    with open(BASE / 'submission/submit_mt_norm_split_b.csv', 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            f.write(f"{i},inset,{0.5 * mt_i2[i] + 0.5 * rf_i[i]:.6f}\n")
        for i in range(50000):
            f.write(f"{i},outset,{0.8 * mt_o[i] + 0.2 * rf_o[i]:.6f}\n")
    print('saved submit_mt_norm_split_b.csv', flush=True)


if __name__ == '__main__':
    main()
