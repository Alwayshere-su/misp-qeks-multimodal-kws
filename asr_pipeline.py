"""
ASR 文本匹配管线
=================
核心思路: query 音频 → Whisper ASR 转写 → 与 enroll 关键词文本匹配打分
- val 模式: 评估 AUC (dev_seen / dev_unseen)
- test 模式: 生成提交 CSV (inset / outset)

用法:
  python asr_pipeline.py --mode eval --model tiny --split dev_seen --limit 500
  python asr_pipeline.py --mode submit --model base --out submission/submit_asr_base.csv
"""
import argparse
import re
import time
import csv
from pathlib import Path

import numpy as np
import torch
import whisper
import soundfile as sf
from sklearn.metrics import roc_auc_score

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')
LOCAL_TINY = BASE / '辅助资源/pretrained_feature_models/whisper/ckps/tiny.pt'


def norm(s: str) -> str:
    """归一化: 小写 + 只保留字母数字与撇号"""
    return re.sub(r"[^a-z0-9' ]", '', (s or '').lower()).strip()


def edit_dist(a: str, b: str) -> int:
    """Levenshtein 编辑距离 (只算小写字母)"""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[lb]


def match_score(enroll: str, asr_text: str, mode: str = 'edit') -> float:
    """
    匹配分数: enroll 关键词 vs ASR 转写
    - exact: 0/1 (完全匹配或子串包含)
    - edit:  连续分数 (词级编辑距离相似度, 容忍 ASR 拼写错误)
    - v2:    stem + 停用词 + 词级编辑距离 (更鲁棒)
    """
    e = norm(enroll)
    q = norm(asr_text)
    if not e or not q:
        return 0.0
    if e == q or e in q or q in e:
        return 1.0
    if mode == 'exact':
        return 0.0
    if mode == 'v2':
        return match_score_v2(e, q)
    # 词级编辑距离相似度
    best = 0.0
    for w in q.split():
        if not w:
            continue
        d = edit_dist(e, w)
        sim = 1.0 - d / max(len(e), len(w), 1)
        if sim > best:
            best = sim
    return max(0.0, best)


STOPWORDS = set("""a an the and or of to in is it this that for with on at by from
as are was were be been being have has had do does did not no so but if then than
you your i we they he she its our their""".split())

try:
    from nltk.stem import PorterStemmer
    _ps = PorterStemmer()
except Exception:
    _ps = None


def _stem(w: str) -> str:
    if _ps is not None:
        return _ps.stem(w)
    w2 = w.rstrip('s')
    for suf in ('ing', 'ed', 'es'):
        if w2.endswith(suf) and len(w2) > len(suf) + 2:
            w2 = w2[:-len(suf)]
            break
    return w2


def match_score_v2(e: str, q: str) -> float:
    """e/q 已归一化"""
    # 去停用词后的词级包含
    qw = [w for w in q.split() if w not in STOPWORDS]
    qjoined = ' '.join(qw)
    if e in qjoined or qjoined in e:
        return 0.99
    es = _stem(e)
    if es == _stem(qjoined) or es == _stem(q):
        return 0.98
    # 词级编辑距离 (stem 后)
    best = 0.0
    for w in (qw or q.split()):
        ws = _stem(w)
        d = edit_dist(es, ws)
        sim = 1.0 - d / max(len(es), len(ws), 1)
        if sim > best:
            best = sim
    # 原词编辑距离兜底
    for w in (qw or q.split()):
        d = edit_dist(e, w)
        sim = 1.0 - d / max(len(e), len(w), 1)
        if sim > best:
            best = sim
    return max(0.0, best)


def load_model(name: str):
    if name in ('tiny', 'base', 'small', 'medium', 'large'):
        if name == 'tiny':
            return whisper.load_model(str(LOCAL_TINY))
        return whisper.load_model(name)
    return whisper.load_model(name)


def asr_batch(model, wav_paths, opt, batch_size=16):
    """批量转写, 返回文本列表"""
    texts = []
    for i in range(0, len(wav_paths), batch_size):
        chunk = wav_paths[i:i + batch_size]
        mels = []
        for p in chunk:
            d, sr = sf.read(str(p), dtype='float32')
            a = whisper.pad_or_trim(torch.from_numpy(d))
            mels.append(whisper.log_mel_spectrogram(a))
        mel_batch = torch.stack(mels).to(model.device)
        results = whisper.decode(model, mel_batch, opt)
        if not isinstance(results, list):
            results = [results]
        texts.extend([r.text for r in results])
    return texts


def load_val_labels(split):
    labels = {}
    for f in (BASE / f'val/{split}/label').glob('*.txt'):
        parts = f.read_text(encoding='utf-8').strip().split('\t')
        labels[f.stem] = {p.split(':', 1)[0]: p.split(':', 1)[1] for p in parts}
    return labels


def eval_split(model, split, limit=None, batch_size=16, mode='edit', modes=None):
    labels = load_val_labels(split)
    items = list(labels.items())
    if limit:
        items = items[:limit]
    if modes is None:
        modes = [mode]
    pids = [pid for pid, _ in items]
    wavs = [str(BASE / f'val/{split}/wav/{pid}_query.wav') for pid in pids]
    t0 = time.time()
    texts = asr_batch(model, wavs, opt, batch_size)
    y = []
    scores = {m: [] for m in modes}
    for (pid, d), txt in zip(items, texts):
        y.append(1 if d['label'] == '1' else 0)
        for m in modes:
            scores[m].append(match_score(d['enrollment'], txt, m))
    for m in modes:
        auc = roc_auc_score(y, scores[m]) if len(set(y)) > 1 else float('nan')
        pos_hit = sum(1 for a, b in zip(y, scores[m]) if a == 1 and b >= 0.999)
        neg_rej = sum(1 for a, b in zip(y, scores[m]) if a == 0 and b <= 0.001)
        print(f'[{split}] mode={m} AUC={auc:.6f} n={len(y)} pos={sum(y)} '
              f'pos_hit={pos_hit}/{sum(y)} neg_rej={neg_rej}/{len(y) - sum(y)} '
              f'time={time.time() - t0:.0f}s', flush=True)
    return {m: (roc_auc_score(y, scores[m]) if len(set(y)) > 1 else float('nan'))
            for m in modes}


def make_submission(model, opt, batch_size=16, out_path='submission/submit_asr.csv',
                    match_inset='edit', match_outset='edit'):
    import pickle
    cache_path = str(Path(out_path).with_suffix('.pkl'))
    texts_cache = {}
    if Path(cache_path).exists():
        try:
            texts_cache = pickle.load(open(cache_path, 'rb'))
            print(f'缓存加载: {len(texts_cache)} 条已转写', flush=True)
        except Exception:
            texts_cache = {}
    rows = []
    for type_, folder, match in [
        ('inset', 'eval_seen_unlabeled', match_inset),
        ('outset', 'eval_unseen_unlabeled', match_outset),
    ]:
        csv_path = BASE / 'test' / ('eval_seen_enroll_text.csv' if type_ == 'inset'
                                    else 'eval_unseen_enroll_text.csv')
        enroll_map = {}
        with open(csv_path, encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                enroll_map[r['id']] = r['enroll_txt']
        pids = sorted(enroll_map.keys())
        todo = [pid for pid in pids if pid not in texts_cache]
        print(f'[{type_}] 待转写 {len(todo)}/{len(pids)} queries...', flush=True)
        t0 = time.time()
        if todo:
            wavs = [str(BASE / f'test/{folder}/wav/{pid}_query.wav') for pid in todo]
            texts = asr_batch(model, wavs, opt, batch_size)
            texts_cache.update(zip(todo, texts))
            with open(cache_path, 'wb') as f:
                pickle.dump(texts_cache, f)
        for pid in pids:
            s = match_score(enroll_map[pid], texts_cache[pid], match)
            idx = int(pid.split('_')[1]) - 1
            rows.append((idx, type_, s))
        print(f'[{type_}] done {time.time() - t0:.0f}s', flush=True)
    rows.sort(key=lambda r: (r[1] == 'outset', r[0]))
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'type', 'prediction'])
        w.writerows(rows)
    print(f'saved {out_path} ({len(rows)} rows)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['eval', 'submit'], default='eval')
    ap.add_argument('--model', default='tiny')
    ap.add_argument('--split', default='dev_seen')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--match', default='edit', choices=['exact', 'edit', 'v2'])
    ap.add_argument('--modes', default=None,
                    help='逗号分隔的多个打分模式, 如 edit,v2 (优先级高于 --match)')
    ap.add_argument('--out', default='submission/submit_asr.csv')
    ap.add_argument('--match-inset', default='edit', choices=['exact', 'edit', 'v2'],
                    help='inset(seen) 打分模式')
    ap.add_argument('--match-outset', default='v2', choices=['exact', 'edit', 'v2'],
                    help='outset(unseen) 打分模式')
    ap.add_argument('--fp16', action='store_true')
    args = ap.parse_args()

    model = load_model(args.model)
    opt = whisper.DecodingOptions(language='en', without_timestamps=True,
                                  fp16=args.fp16)
    print(f'model={args.model} device={model.device}', flush=True)

    if args.mode == 'eval':
        modes = args.modes.split(',') if args.modes else [args.match]
        eval_split(model, args.split, args.limit, args.batch, args.match, modes)
    else:
        make_submission(model, opt, args.batch, args.out,
                        args.match_inset, args.match_outset)
