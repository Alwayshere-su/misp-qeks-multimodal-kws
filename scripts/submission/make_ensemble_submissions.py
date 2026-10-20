"""p4 5 折集成对 test 打分 + 与 s2u/full 组合成多种集成提交"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_whisper_p4 import MatcherV4, encode_text

BASE = Path(r"D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_split(model, data, lip, bs=512):
    model.eval()
    pids = sorted(data.keys())
    preds = {}
    for i in range(0, len(pids), bs):
        chunk = pids[i:i + bs]
        ea = torch.stack([torch.tensor(data[p]['enroll_emb'], dtype=torch.float32) for p in chunk]).to(DEVICE)
        qa = torch.stack([torch.tensor(data[p]['query_emb'], dtype=torch.float32) for p in chunk]).to(DEVICE)
        ev = torch.stack([torch.tensor(lip[p]['enroll_video'], dtype=torch.float32) for p in chunk]).to(DEVICE)
        qv = torch.stack([torch.tensor(lip[p]['query_video'], dtype=torch.float32) for p in chunk]).to(DEVICE)
        t = torch.stack([encode_text(data[p]['enroll_txt']) for p in chunk]).to(DEVICE)
        logit = model(ea, qa, ev, qv, t)
        prob = torch.sigmoid(logit).cpu().flatten().tolist()
        for p, pr in zip(chunk, prob):
            preds[p] = pr
    return preds


def rank_norm(preds):
    from scipy.stats import rankdata
    pids = sorted(preds.keys())
    v = np.array([preds[p] for p in pids])
    return dict(zip(pids, rankdata(v) / len(v)))


def write_csv(inset, outset, path):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},inset,{inset.get(pid, 0.5):.6f}\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            f.write(f"{i},outset,{outset.get(pid, 0.5):.6f}\n")
    print(f"  -> {path}")


def load_csv(path):
    import csv
    out = {}
    for r in csv.DictReader(open(path, encoding='utf-8')):
        out.setdefault(r['type'], {})[f"pair_{int(r['id'])+1:06d}"] = float(r['prediction'])
    return out.get('inset', {}), out.get('outset', {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='fold', choices=['fold', 'combo'])
    args = ap.parse_args()

    feats_inset = pickle.load(open(BASE / 'output_whisper/test_inset.pkl', 'rb'))
    feats_outset = pickle.load(open(BASE / 'output_whisper/test_outset.pkl', 'rb'))
    lip_inset = pickle.load(open(BASE / 'output_lip/test_lip_inset.pkl', 'rb'))
    lip_outset = pickle.load(open(BASE / 'output_lip/test_lip_outset.pkl', 'rb'))

    if args.mode == 'fold':
        # 5 折模型集成
        inset_acc = {}
        outset_acc = {}
        for fold in range(5):
            model = MatcherV4().to(DEVICE)
            model.load_state_dict(torch.load(BASE / f'output_p4/best_fold{fold}.pt',
                                             map_location='cpu', weights_only=True))
            model.eval()
            pi = predict_split(model, feats_inset, lip_inset)
            po = predict_split(model, feats_outset, lip_outset)
            for p in pi: inset_acc.setdefault(p, []).append(pi[p])
            for p in po: outset_acc.setdefault(p, []).append(po[p])
            print(f'fold{fold} done', flush=True)
        inset = {p: float(np.mean(v)) for p, v in inset_acc.items()}
        outset = {p: float(np.mean(v)) for p, v in outset_acc.items()}
        write_csv(inset, outset, str(BASE / 'submission/submit_p4_foldavg_test.csv'))

    if args.mode == 'combo':
        # 组合: s2u + foldavg + full (rank 平均)
        from scipy.stats import rankdata
        parts = []
        for name in ['submit_p4_s2u_test.csv', 'submit_p4_foldavg_test.csv',
                     'submit_p4_full.csv', 'submit_p1_aug_test.csv']:
            ii, oo = load_csv(BASE / 'submission' / name)
            parts.append((rank_norm(ii), rank_norm(oo)))
        inset = {p: float(np.mean([pt[0][p] for pt in parts])) for p in parts[0][0]}
        outset = {p: float(np.mean([pt[1][p] for pt in parts])) for p in parts[0][1]}
        write_csv(inset, outset, str(BASE / 'submission/submit_combo4_test.csv'))
        # 两两: s2u + foldavg
        for a, b, nm in [('submit_p4_s2u_test.csv', 'submit_p4_foldavg_test.csv', 'submit_s2u_foldavg.csv')]:
            ii_a, oo_a = load_csv(BASE / 'submission' / a)
            ii_b, oo_b = load_csv(BASE / 'submission' / b)
            inset = {p: 0.5 * (rank_norm(ii_a)[p] + rank_norm(ii_b)[p]) for p in ii_a}
            outset = {p: 0.5 * (rank_norm(oo_a)[p] + rank_norm(oo_b)[p]) for p in oo_a}
            write_csv(inset, outset, str(BASE / 'submission' / nm))


if __name__ == '__main__':
    main()
