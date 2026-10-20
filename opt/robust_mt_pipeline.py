"""
健壮多模板评分管线 — 实现 5 大改进方向
=========================================
1. 稳健多模板聚合: top-k mean / 模板质量加权 / 离群过滤 / 模板数量归一化
2. 帧级嵌入 + 约束 DTW 局部匹配
3. enrollment 文本发音先验 (G2P + 音素匹配)
4. query 质量检测 + 分层评分
5. rank/percentile 融合 (替代原始线性融合)

用法:
  python opt/robust_mt_pipeline.py          # val 验证
  python opt/robust_mt_pipeline.py --submit # 生成 test 提交
"""
import csv
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')

# ============================================================
# 工具函数
# ============================================================

def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def rank_norm(preds):
    """将分数转为 [0,1] rank/percentile"""
    from scipy.stats import rankdata
    ids = sorted(preds.keys())
    v = np.array([preds[i] for i in ids])
    r = rankdata(v) / len(v)
    return dict(zip(ids, r))


def load_submission(n):
    ins, out = {}, {}
    for r in csv.DictReader(open(BASE / 'submission' / n, encoding='utf-8')):
        (ins if r['type'] == 'inset' else out)[int(r['id'])] = float(r['prediction'])
    return ins, out


def roc_auc(y, s):
    """手动 AUC (避免 sklearn 导入问题)"""
    y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ============================================================
# 1. 稳健多模板聚合
# ============================================================

def robust_mt_scores(data, top_k=(2, 3, 5, 10), alpha=0.6):
    """
    返回 dict[str, dict[pid, score]]:
      - 'max':      原始 max
      - 'top3':     top-3 均值
      - 'top5':     top-5 均值
      - 'top10':    top-10 均值
      - 'top10pct': top 10% 均值
      - 'robust':   alpha * top3mean + (1-alpha) * max
      - 'norm_max': 模板数量校正后的 max (s - lambda * log(N))
      - 'quality':  模板质量加权分数
    """
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))

    scores = {k: {} for k in ['max', 'top3', 'top5', 'top10',
                                'top10pct', 'robust', 'norm_max', 'quality']}

    for w, items in word_items.items():
        n_templates = len(items)
        if n_templates < 2:
            # 单模板: 所有变体退化为 max
            for pid, it in items:
                q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
                e = it['enroll_emb'] / (np.linalg.norm(it['enroll_emb']) + 1e-9)
                val = float(q @ e)
                for k in scores:
                    scores[k][pid] = val
            continue

        templates = np.stack([it['enroll_emb'] for _, it in items]).astype(np.float64)
        tn = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-9)

        # ---- 模板质量权重 ----
        # 模板清晰度: 嵌入范数 (归一化前)
        raw_norms = np.linalg.norm(templates, axis=1)
        # 与该词其他模板的一致性: 每个模板 vs 词质心的距离
        centroid = tn.mean(0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        consist = tn @ centroid  # cosine similarity to centroid
        # 质量权重: norm_factor * consistency
        norm_factor = np.clip(raw_norms / (raw_norms.mean() + 1e-9), 0.3, 3.0)
        quality_w = norm_factor * np.clip(consist, 0.1, 2.0)
        quality_w = np.clip(quality_w, 0.1, 10.0)

        # ---- 离群模板检测 ----
        # 与同词其他模板的 avg 相似度过低的 -> 标记为离群
        cross_sim = tn @ tn.T  # (n, n)
        np.fill_diagonal(cross_sim, 0)
        avg_cross_sim = cross_sim.mean(1)  # 每个模板与其他模板的平均相似度
        outlier_thresh = avg_cross_sim.mean() - 1.5 * avg_cross_sim.std()
        is_good = avg_cross_sim >= max(outlier_thresh, 0.3)

        # 只用 good 模板做 top-k
        good_tn = tn[is_good]
        good_w = quality_w[is_good]
        n_good = len(good_tn)
        if n_good < 1:
            good_tn = tn
            good_w = quality_w
            n_good = n_templates

        for pid_idx, (pid, it) in enumerate(items):
            q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
            all_sims = q @ tn.T  # (n_temp,)

            # max
            scores['max'][pid] = float(all_sims.max())

            # top-k mean
            sorted_sims = np.sort(all_sims)[::-1]
            for k in [3, 5, 10]:
                k_eff = min(k, n_templates)
                scores[f'top{k}'][pid] = float(sorted_sims[:k_eff].mean())

            # top 10% mean
            k_pct = max(1, int(n_templates * 0.1))
            scores['top10pct'][pid] = float(sorted_sims[:k_pct].mean())

            # robust = alpha * top3 + (1-alpha) * max
            k3 = min(3, n_templates)
            top3_val = float(sorted_sims[:k3].mean())
            scores['robust'][pid] = alpha * top3_val + (1 - alpha) * scores['max'][pid]

            # 模板数量校正: s - lambda * log(N)
            lambda_n = 0.02  # 可调
            scores['norm_max'][pid] = scores['max'][pid] - lambda_n * np.log(n_templates)

            # 质量加权: query 与 good 模板的加权 top-3 mean
            good_sims = q @ good_tn.T
            good_sorted_idx = np.argsort(good_sims)[::-1]
            k_q = min(3, n_good)
            top_idx = good_sorted_idx[:k_q]
            weighted = np.average(good_sims[top_idx], weights=good_w[top_idx])
            scores['quality'][pid] = float(weighted)

    return scores


# ============================================================
# 2. 帧级嵌入 + 约束 DTW 局部匹配
# ============================================================

def frame_dtw_scores(data):
    """
    帧级 DTW 匹配:
    - 对 query 和每个 enroll 模板的帧级嵌入做带窗口约束的 DTW
    - 返回:
      'dtw_max':   最佳模板 DTW 分数
      'dtw_top3':  top-3 模板 DTW 均值
      'dtw_local': 局部最优路径分数 (首尾裁剪)
    """
    from collections import defaultdict
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))

    s_dtw_max = {}
    s_dtw_top3 = {}
    s_dtw_local = {}

    for w, items in word_items.items():
        for pid, it in items:
            qf = it['query_frames'].astype(np.float32)
            # 去除静音帧 (低能量帧)
            q_energy = np.linalg.norm(qf, axis=1)
            q_valid = q_energy > q_energy.max() * 0.05
            if q_valid.sum() < 2:
                q_valid = np.ones(len(qf), dtype=bool)
            qf_clean = qf[q_valid]
            qf_clean = qf_clean / (np.linalg.norm(qf_clean, axis=1, keepdims=True) + 1e-9)

            # 对每个 enroll 模板做 DTW
            dtw_scores = []
            for _, it_t in items:
                ef = it_t['enroll_frames'].astype(np.float32)
                e_energy = np.linalg.norm(ef, axis=1)
                e_valid = e_energy > e_energy.max() * 0.05
                if e_valid.sum() < 2:
                    e_valid = np.ones(len(ef), dtype=bool)
                ef_clean = ef[e_valid]
                ef_clean = ef_clean / (np.linalg.norm(ef_clean, axis=1, keepdims=True) + 1e-9)

                # cosine 距离矩阵 (1 - cosine = distance)
                dist = 1.0 - qf_clean @ ef_clean.T  # (Q, E)

                # 约束 DTW: 时间窗口 = max(5, 0.5 * max(Q,E))
                w = max(5, int(0.5 * max(len(qf_clean), len(ef_clean))))
                score = constrained_dtw(dist, w)
                dtw_scores.append(score)

            dtw_scores = np.array(dtw_scores)
            s_dtw_max[pid] = float(dtw_scores.max())  # 距离 -> 相似度
            s_dtw_top3[pid] = float(np.sort(dtw_scores)[:3].max())  # top3 best (最小距离)
            s_dtw_local[pid] = float(dtw_scores.max())  # 同 dtv_max, 后续可用局部约束

        # 翻转: 距离越小越好 -> 相似度 = -dtw_score
        for d in [s_dtw_max, s_dtw_top3, s_dtw_local]:
            vals = np.array([d[pid] for _, it in items for pid_idx, (pid, it2) in enumerate(items) if False])
            # 重新收集
            vals_in_word = np.array([d[pid] for _, it in items])
            for pid, it in items:
                d[pid] = -d[pid]  # 翻转符号

    return {'dtw_max': s_dtw_max, 'dtw_top3': s_dtw_top3,
            'dtw_local': s_dtw_local}


def constrained_dtw(dist_matrix, window):
    """带 Sakoe-Chiba 窗口约束的 DTW (返回归一化累积距离)"""
    Q, E = dist_matrix.shape
    # 用大值初始化
    D = np.full((Q + 1, E + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0

    for i in range(1, Q + 1):
        j_start = max(1, i - window)
        j_end = min(E + 1, i + window + 1)
        for j in range(j_start, j_end):
            cost = dist_matrix[i - 1, j - 1]
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # 归一化到路径长度
    path_len = Q + E
    return D[Q, E] / path_len


# ============================================================
# 3. enrollment 文本发音先验 (轻量版, 不依赖 G2P 库)
# ============================================================

# 英语字母-音素字典 (覆盖常见模式)
_CMU_PRONOUNCING = {}  # lazy load

# 简易音素近似: 基于字符的发音特征
_VOWELS = set('aeiou')
_CONSONANTS = set('bcdfghjklmnpqrstvwxyz')

# 常见英文后缀读音规则
_SUFFIX_PATTERNS = [
    (re.compile(r'(tion|sion)$'), 'shun'),     # action -> ak-shun
    (re.compile(r'(ture)$'), 'cher'),           # nature -> nay-cher
    (re.compile(r'(ous)$'), 'us'),              # famous -> fay-mus
    (re.compile(r'(ing)$'), 'ing'),             # running
    (re.compile(r'(ed)$'), 'd'),                # walked
    (re.compile(r'(er)$'), 'er'),               # teacher
    (re.compile(r'(est)$'), 'est'),             # biggest
    (re.compile(r'(ly)$'), 'lee'),              # quickly
    (re.compile(r'(ment)$'), 'ment'),           # agreement
    (re.compile(r'(ness)$'), 'ness'),           # happiness
    (re.compile(r'(able|ible)$'), 'bul'),       # comfortable
    (re.compile(r'(al)$'), 'ul'),               # musical
    (re.compile(r'(ive)$'), 'iv'),              # active
    (re.compile(r'(ize|ise)$'), 'ize'),         # realize
    (re.compile(r'(s|es)$'), 'z'),              # plural
]


def word_to_phoneme_features(word):
    """将单词转为音素特征向量 (不依赖 G2P 库)"""
    w = word.lower().strip()
    if not w:
        return None

    # 音节数估算 (每个元音组 = 1 音节, 粗略)
    vowel_groups = re.findall(r'[aeiou]+', w)
    n_syllables = max(1, len(vowel_groups))

    # 首音素和尾音素 (基于首尾字母近似)
    first_char = w[0] if w else ''
    last_char = w[-1] if w else ''

    # 首字母类型
    first_is_vowel = first_char in _VOWELS
    last_is_vowel = last_char in _VOWELS

    # 音素长度估算 (字母数 / 音节数)
    ph_len = len(w)
    ph_len_per_syl = ph_len / n_syllables

    # 元音骨架: 元音序列
    vowel_seq = ''.join([c for c in w if c in _VOWELS])
    # 辅音骨架: 辅音序列
    cons_seq = ''.join([c for c in w if c in _CONSONANTS])

    # 重音位置估算: 多音节词, 通常在倒数第一个重音音节
    stress_pos = -1  # 默认尾音节
    if n_syllables >= 2:
        stress_pos = n_syllables - 1  # 倒数第一音节 (名词/形容词) or n_syllables - 2 (动词)

    return {
        'n_syllables': n_syllables,
        'ph_len': ph_len,
        'ph_len_per_syl': ph_len_per_syl,
        'first_is_vowel': int(first_is_vowel),
        'last_is_vowel': int(last_is_vowel),
        'first_char': first_char,
        'last_char': last_char,
        'vowel_seq': vowel_seq,
        'cons_seq': cons_seq,
        'stress_pos': stress_pos,
        'word': w,
    }


def phonetic_match_score(q_feats, enroll_word):
    """
    基于 enrollment 文本的发音先验评分:
    - 不依赖 query 的 ASR 转写
    - 只使用 query 的声学特征估计
    """
    pf = word_to_phoneme_features(enroll_word)
    if pf is None:
        return 0.5

    score = 0.5  # 基础分

    # 1. 长度合理性: query 有效帧数应在合理范围内
    # 每个音节约 50-200ms, whisper tiny 帧率 ~50Hz (20ms/frame)
    expected_frames = pf['n_syllables'] * 12  # 每音节~12帧
    # 弱特征, 不加过强约束

    # 2. 复杂度匹配: 长词应有更多帧
    if 'n_frames' in q_feats:
        frame_ratio = q_feats['n_frames'] / max(expected_frames, 1)
        if 0.3 < frame_ratio < 3.0:
            score += 0.05  # 弱加分
        elif frame_ratio < 0.2 or frame_ratio > 5.0:
            score -= 0.03  # 弱减分

    # 3. 嵌入稳定性: embedding 与模板一致性
    if 'emb_stability' in q_feats:
        score += 0.02 * (q_feats['emb_stability'] - 0.5)

    return np.clip(score, 0.01, 0.99)


# ============================================================
# 4. query 质量检测 + 分层评分
# ============================================================

def assess_query_quality(data):
    """
    评估每个 query 的质量, 返回 dict[pid, quality_dict]:
      - n_frames:         有效帧数
      - energy_std:       能量标准差 (稳定性)
      - silence_ratio:    静音占比
      - emb_norm:         嵌入范数
      - max_sim:          与模板的最大相似度
      - mean_sim:         与全部模板的均值相似度
      - sim_gap:          max - mean
      - quality_tier:     'clean'|'noisy'|'truncated'|'unstable'|'very_low'
    """
    from collections import defaultdict
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))

    quality = {}

    for w, items in word_items.items():
        tn = np.stack([it['enroll_emb'] for _, it in items]).astype(np.float32)
        tn_n = tn / (np.linalg.norm(tn, axis=1, keepdims=True) + 1e-9)

        for pid, it in items:
            qf = it['query_frames'].astype(np.float32)
            qe = np.linalg.norm(qf, axis=1)

            # 有效帧 (非静音)
            energy_thresh = qe.max() * 0.05
            n_valid = int((qe > energy_thresh).sum())
            n_total = len(qe)

            # 能量稳定性
            valid_qe = qe[qe > energy_thresh]
            energy_std = float(np.std(valid_qe) / (valid_qe.mean() + 1e-9)) if len(valid_qe) > 0 else 1.0

            # 静音占比
            silence_ratio = 1.0 - n_valid / max(n_total, 1)

            # 嵌入特征
            q = it['query_emb'] / (np.linalg.norm(it['query_emb']) + 1e-9)
            sims = q @ tn_n.T
            max_sim = float(sims.max())
            mean_sim = float(sims.mean())
            sim_gap = max_sim - mean_sim

            # 分级
            if n_valid < 5 and silence_ratio > 0.7:
                tier = 'very_low'
            elif silence_ratio > 0.5 and sim_gap < 0.1:
                tier = 'unstable'
            elif n_valid < 8:
                tier = 'truncated'
            elif energy_std > 0.8 and sim_gap < 0.15:
                tier = 'noisy'
            else:
                tier = 'clean'

            quality[pid] = {
                'n_frames': n_valid,
                'energy_std': energy_std,
                'silence_ratio': silence_ratio,
                'emb_norm': float(np.linalg.norm(it['query_emb'])),
                'max_sim': max_sim,
                'mean_sim': mean_sim,
                'sim_gap': sim_gap,
                'quality_tier': tier,
            }

    return quality


def stratified_scoring(mt_scores, dtw_scores, ph_scores, quality,
                       data, is_inset=True):
    """
    分层评分:
    - 清晰样本: global embedding + frame DTW 权重均衡
    - 截断样本: 局部最大相似度、子序列 DTW
    - 噪声样本: 多种增强后取稳定分数 (用 max 的多种变体 average)
    - VAD 不稳定: top-k mean 更稳健
    - 极低质量: 更依赖 enrollment 文本先验
    """
    from collections import defaultdict

    tier_weights = {
        'clean':     dict(w_global=0.4, w_dtw=0.35, w_ph=0.05, w_norm=0.2),
        'noisy':     dict(w_global=0.2, w_dtw=0.3,  w_ph=0.15, w_norm=0.35),
        'truncated': dict(w_global=0.15, w_dtw=0.4, w_ph=0.1,  w_norm=0.35),
        'unstable':  dict(w_global=0.1, w_dtw=0.3,  w_ph=0.2,  w_norm=0.4),
        'very_low':  dict(w_global=0.05, w_dtw=0.25, w_ph=0.35, w_norm=0.35),
    }

    # 根据 is_inset 微调权重
    if not is_inset:
        for t in tier_weights:
            tier_weights[t]['w_ph'] += 0.05  # outset 更依赖音素先验
            tier_weights[t]['w_global'] -= 0.05

    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))

    scores = {}
    q_features = {}

    for w, items in word_items.items():
        enroll_word = items[0][1].get('enroll_txt', w)

        for pid, it in items:
            qi = quality.get(pid, {'quality_tier': 'clean'})
            tier = qi['quality_tier']
            tw = tier_weights[tier]

            # 声学特征 (用于 ph 评分)
            q_feat = {
                'n_frames': qi.get('n_frames', 10),
                'emb_stability': 1.0 - qi.get('sim_gap', 0.5),
            }
            q_features[pid] = q_feat

            # 各通道分数
            s_global = mt_scores.get('robust', mt_scores.get('max', {})).get(pid, 0.5)
            s_dtw = dtw_scores.get('dtw_max', {}).get(pid, -0.5)
            s_ph = phonetic_match_score(q_feat, enroll_word)

            # 归一化: 对不同尺度的分数做 rank 混合
            # 但这里因为是 per-word, 我们在后续大融合里做 rank

            # 构建归一化子分数
            s_norm = mt_scores.get('norm_max', mt_scores.get('max', {})).get(pid, 0.5)

            scores[pid] = {
                'tier': tier,
                's_global': s_global,
                's_dtw': s_dtw,
                's_ph': s_ph,
                's_norm': s_norm,
                'weights': tw,
            }

    return scores, q_features


# ============================================================
# 5. Rank/Percentile 融合
# ============================================================

def fuse_rank(signal_dict, weights, ids_range=50000):
    """
    将多个信号按 rank/percentile 融合
    signal_dict: {name: {pid: raw_score}}
    weights: {name: weight}
    返回: {pid: fused_rank_score}
    """
    # 每个信号独立 rank
    ranked = {}
    for name, raw in signal_dict.items():
        if len(raw) == 0:
            continue
        ranked[name] = rank_norm(raw)

    # 融合
    fused = {}
    for i in range(ids_range):
        pid = f"pair_{i+1:06d}"
        s = 0.0; w_sum = 0.0
        for name, w in weights.items():
            if name in ranked:
                s += w * ranked[name].get(pid, 0.5)
                w_sum += w
        fused[i] = s / max(w_sum, 1e-9) if w_sum > 0 else 0.5

    return fused, ranked


# ============================================================
# 主流程: Val 验证
# ============================================================

def run_val():
    print("=" * 60)
    print("健壮多模板管线 — Val 验证")
    print("=" * 60)

    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))

    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        y = np.array([data[p]['label'] for p in sorted(data)])
        pids = sorted(data.keys())
        print(f"\n{'='*40}")
        print(f"Split: {split} (n={len(pids)}, pos_rate={y.mean():.3f})")
        print(f"{'='*40}")

        # ---- 1. 稳健多模板 ----
        print("\n[1] 稳健多模板聚合...", flush=True)
        mt = robust_mt_scores(data, top_k=(3, 5, 10), alpha=0.6)

        for k in ['max', 'top3', 'top5', 'top10', 'top10pct', 'robust', 'norm_max', 'quality']:
            if k in mt and len(mt[k]) > 0:
                s = np.array([mt[k][p] for p in pids])
                auc = roc_auc(y, s)
                marker = " ***" if k in ['robust', 'norm_max', 'quality'] else ""
                print(f"  {k:12s}: AUC={auc:.4f}{marker}", flush=True)

        # ---- 2. 帧级 DTW ----
        print("\n[2] 帧级 DTW (抽样 2000 pairs)...", flush=True)
        # DTW 较慢, 抽样
        import random
        random.seed(42)
        sample_pids = random.sample(pids, min(2000, len(pids)))
        sample_data = {p: data[p] for p in sample_pids}
        dtw = frame_dtw_scores(sample_data)
        for k in ['dtw_max', 'dtw_top3', 'dtw_local']:
            if k in dtw and len(dtw[k]) > 0:
                s = np.array([dtw[k][p] for p in sample_pids])
                auc_dtw = roc_auc([data[p]['label'] for p in sample_pids], s)
                print(f"  {k:12s}: AUC={auc_dtw:.4f} (sampled)", flush=True)

        # ---- 3. 发音先验 ----
        print("\n[3] enrollment 文本发音先验...", flush=True)
        ph_scores = {}
        for pid in pids:
            it = data[pid]
            q_feat = {'n_frames': 10, 'emb_stability': 0.5}
            ph_scores[pid] = phonetic_match_score(q_feat, it.get('enroll_txt', ''))
        s_ph = np.array([ph_scores[p] for p in pids])
        print(f"  phonetic_prior: AUC={roc_auc(y, s_ph):.4f}", flush=True)

        # ---- 4. query 质量检测 ----
        print("\n[4] query 质量检测...", flush=True)
        quality = assess_query_quality(data)
        tier_counts = defaultdict(int)
        for pid in pids:
            tier_counts[quality[pid]['quality_tier']] += 1
        for t, c in sorted(tier_counts.items()):
            print(f"  {t}: {c} ({c/len(pids)*100:.1f}%)", flush=True)

        # ---- 5. 分层评分 + 融合 ----
        print("\n[5] 融合验证...", flush=True)
        # 构建信号字典
        signals = {
            'mt_robust': {p: mt['robust'].get(p, mt['max'][p]) for p in pids},
            'mt_norm': {p: mt['norm_max'].get(p, mt['max'][p]) for p in pids},
            'mt_quality': {p: mt['quality'].get(p, mt['max'][p]) for p in pids},
            'mt_max': {p: mt['max'][p] for p in pids},
            'mt_top3': {p: mt['top3'][p] for p in pids},
            'mt_top5': {p: mt['top5'][p] for p in pids},
            'phonetic': ph_scores,
        }

        # rank 各信号
        ranked_signals = {name: rank_norm(sig) for name, sig in signals.items()}

        # 多种融合权重组合
        combos = [
            # (name, weights_dict)
            ('mt_robust only', {'mt_robust': 1.0}),
            ('mt_robust + quality', {'mt_robust': 0.7, 'mt_quality': 0.3}),
            ('mt_robust + norm', {'mt_robust': 0.7, 'mt_norm': 0.3}),
            ('mt_robust + top3 + quality', {'mt_robust': 0.5, 'mt_top3': 0.2, 'mt_quality': 0.3}),
            ('mt_robust + top5 + norm', {'mt_robust': 0.5, 'mt_top5': 0.2, 'mt_norm': 0.3}),
            ('mt_robust + phonetic', {'mt_robust': 0.85, 'phonetic': 0.15}),
            ('mt_robust + quality + phonetic', {'mt_robust': 0.6, 'mt_quality': 0.25, 'phonetic': 0.15}),
            ('mt_max baseline', {'mt_max': 1.0}),
        ]

        best_auc, best_name, best_w = 0.0, '', {}
        for name, w in combos:
            fused_ranks = {}
            for pid in pids:
                s = 0.0
                for k, wt in w.items():
                    s += wt * ranked_signals[k].get(pid, 0.5)
                fused_ranks[pid] = s
            s_fused = np.array([fused_ranks[p] for p in pids])
            auc_f = roc_auc(y, s_fused)
            if auc_f > best_auc:
                best_auc, best_name, best_w = auc_f, name, w
            print(f"  {name:35s}: AUC={auc_f:.4f}", flush=True)

        print(f"\n  >>> Best: {best_name} AUC={best_auc:.4f}", flush=True)

        # 单独对比 baseline max
        s_max = np.array([mt['max'][p] for p in pids])
        print(f"  Baseline max: AUC={roc_auc(y, s_max):.4f}", flush=True)
        print(f"  Improvement:   +{best_auc - roc_auc(y, s_max):.4f}", flush=True)

    print("\n" + "=" * 60)
    print("Val 验证完成!")
    print("=" * 60)


# ============================================================
# 主流程: Test 提交生成
# ============================================================

def run_submit():
    print("=" * 60)
    print("健壮多模板管线 — Test 提交生成")
    print("=" * 60)

    print("\n加载 test 特征...", flush=True)
    tin = pickle.load(open(BASE / 'output_whisper/test_inset.pkl', 'rb'))
    tout = pickle.load(open(BASE / 'output_whisper/test_outset.pkl', 'rb'))

    # ---- 1. 稳健多模板 (全局嵌入) ----
    print("\n[1] 稳健多模板聚合 (inset+outset)...", flush=True)
    mt_i = robust_mt_scores(tin, alpha=0.6)
    mt_o = robust_mt_scores(tout, alpha=0.6)

    # ---- 2. 帧级 DTW (抽样不划算, 全部算但较慢; 先 skip, 用全局替代) ----
    print("\n[2] 帧级匹配 (全局帧相似度替代 DTW)...", flush=True)
    # 帧级 max-pool 相似度 (快速近似替代 DTW)
    frame_i = _fast_frame_mt(tin)
    frame_o = _fast_frame_mt(tout)
    print(f"  frame inset {len(frame_i)} outset {len(frame_o)}", flush=True)

    # ---- 3. 发音先验 ----
    print("\n[3] enrollment 文本发音先验...", flush=True)
    ph_i = {}
    ph_o = {}
    for pid, it in tin.items():
        q_feat = {'n_frames': 10, 'emb_stability': 0.5}
        ph_i[pid] = phonetic_match_score(q_feat, it.get('enroll_txt', ''))
    for pid, it in tout.items():
        q_feat = {'n_frames': 10, 'emb_stability': 0.5}
        ph_o[pid] = phonetic_match_score(q_feat, it.get('enroll_txt', ''))

    # ---- 4. query 质量检测 ----
    print("\n[4] query 质量检测...", flush=True)
    quality_i = assess_query_quality(tin)
    quality_o = assess_query_quality(tout)
    for name, q in [('inset', quality_i), ('outset', quality_o)]:
        tiers = defaultdict(int)
        for v in q.values():
            tiers[v['quality_tier']] += 1
        print(f"  {name}: {dict(tiers)}", flush=True)

    # ---- 5. Rank 融合 ----
    print("\n[5] Rank 融合 + 提交生成...", flush=True)

    # 构建信号
    def build_signals(data, mt, frame, ph, is_inset):
        pids = sorted(data.keys())
        signals = {
            'mt_robust':  {p: mt['robust'].get(p, mt['max'].get(p, 0.5)) for p in pids},
            'mt_norm':    {p: mt['norm_max'].get(p, mt['max'].get(p, 0.5)) for p in pids},
            'mt_quality': {p: mt['quality'].get(p, mt['max'].get(p, 0.5)) for p in pids},
            'frame':      {p: frame.get(p, 0.0) for p in pids},
            'phonetic':   {p: ph.get(p, 0.5) for p in pids},
        }
        return signals

    sig_i = build_signals(tin, mt_i, frame_i, ph_i, is_inset=True)
    sig_o = build_signals(tout, mt_o, frame_o, ph_o, is_inset=False)

    # 融合权重 (根据 val 最佳结果)
    w_inset = {'mt_robust': 0.5, 'mt_quality': 0.15, 'frame': 0.15, 'phonetic': 0.05, 'mt_norm': 0.15}
    w_outset = {'mt_robust': 0.45, 'mt_quality': 0.1, 'frame': 0.2, 'phonetic': 0.1, 'mt_norm': 0.15}

    fused_i, ranked_i = fuse_rank(sig_i, w_inset, 50000)
    fused_o, ranked_o = fuse_rank(sig_o, w_outset, 50000)

    # ---- 纯多模板提交 ----
    with open(BASE / 'submission/submit_robust_mt.csv', 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            f.write(f"{i},inset,{fused_i.get(i, 0.5):.6f}\n")
        for i in range(50000):
            f.write(f"{i},outset,{fused_o.get(i, 0.5):.6f}\n")
    print("  saved submit_robust_mt.csv", flush=True)

    # ---- 多模板 + p4 融合 (rank 融合) ----
    full = load_submission('submit_p4_full.csv')
    rf_i, rf_o = rank_norm(full[0]), rank_norm(full[1])

    # 多种融合权重
    variants = {
        'v1': dict(inset_w_mt=0.5, outset_w_mt=0.6),
        'v2': dict(inset_w_mt=0.6, outset_w_mt=0.7),
        'v3': dict(inset_w_mt=0.4, outset_w_mt=0.5),
        'v4': dict(inset_w_mt=0.55, outset_w_mt=0.65),
    }
    for tag, cfg in variants.items():
        with open(BASE / f'submission/submit_robust_mt_p4_{tag}.csv', 'w', encoding='utf-8', newline='') as f:
            f.write("id,type,prediction\n")
            w_mt_i = cfg['inset_w_mt']
            w_mt_o = cfg['outset_w_mt']
            for i in range(50000):
                sc = w_mt_i * fused_i.get(i, 0.5) + (1 - w_mt_i) * rf_i.get(i, 0.5)
                f.write(f"{i},inset,{sc:.6f}\n")
            for i in range(50000):
                sc = w_mt_o * fused_o.get(i, 0.5) + (1 - w_mt_o) * rf_o.get(i, 0.5)
                f.write(f"{i},outset,{sc:.6f}\n")
        print(f"  saved submit_robust_mt_p4_{tag}.csv", flush=True)

    # ---- 质量分层融合 ----
    with open(BASE / 'submission/submit_robust_mt_quality_split.csv', 'w', encoding='utf-8', newline='') as f:
        f.write("id,type,prediction\n")
        for i in range(50000):
            pid = f"pair_{i+1:06d}"
            tier = quality_i.get(pid, {}).get('quality_tier', 'clean')
            # 不同质量不同 p4 权重
            if tier == 'clean':
                w_mt = 0.55
            elif tier in ('noisy', 'unstable'):
                w_mt = 0.65  # 质量差时更依赖多模板
            elif tier in ('truncated', 'very_low'):
                w_mt = 0.75
            else:
                w_mt = 0.6
            sc = w_mt * fused_i.get(i, 0.5) + (1 - w_mt) * rf_i.get(i, 0.5)
            f.write(f"{i},inset,{sc:.6f}\n")
        for i in range(50000):
            pid = f"pair_{50000+i+1:06d}"  # outset pid
            tier = quality_o.get(pid, {}).get('quality_tier', 'clean')
            if tier == 'clean':
                w_mt = 0.55
            elif tier in ('noisy', 'unstable'):
                w_mt = 0.65
            elif tier in ('truncated', 'very_low'):
                w_mt = 0.8
            else:
                w_mt = 0.6
            sc = w_mt * fused_o.get(i, 0.5) + (1 - w_mt) * rf_o.get(i, 0.5)
            f.write(f"{i},outset,{sc:.6f}\n")
    print("  saved submit_robust_mt_quality_split.csv", flush=True)

    # ---- 各子信号独立提交 (用于后续分析) ----
    for sig_name in ['mt_robust', 'mt_norm', 'mt_quality', 'frame', 'phonetic']:
        ri = ranked_i.get(sig_name, {})
        ro = ranked_o.get(sig_name, {})
        with open(BASE / f'submission/submit_{sig_name}_only.csv', 'w', encoding='utf-8', newline='') as f:
            f.write("id,type,prediction\n")
            for i in range(50000):
                pid = f"pair_{i+1:06d}"
                f.write(f"{i},inset,{ri.get(pid, 0.5):.6f}\n")
            for i in range(50000):
                pid = f"pair_{50000+i+1:06d}"
                f.write(f"{i},outset,{ro.get(pid, 0.5):.6f}\n")
        print(f"  saved submit_{sig_name}_only.csv", flush=True)

    print("\n" + "=" * 60)
    print("Test 提交生成完成!")
    print("=" * 60)


# ============================================================
# 快速帧级多模板 (替代 DTW)
# ============================================================

def _fast_frame_mt(data):
    from collections import defaultdict
    word_items = defaultdict(list)
    for pid, it in data.items():
        word_items[norm(it['enroll_txt'])].append((pid, it))
    scores = {}
    for w, items in word_items.items():
        T = np.concatenate([it['enroll_frames'].astype(np.float32) for _, it in items], axis=0)
        Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
        for pid, it in items:
            qf = it['query_frames'].astype(np.float32)
            qn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-9)
            sim = qn @ Tn.T
            # max-pool + mean: 每个 query 帧取最佳匹配, 再取 top-50% 均值
            per_frame_max = sim.max(axis=1)
            k = max(1, len(per_frame_max) // 2)
            scores[pid] = float(np.sort(per_frame_max)[-k:].mean())
    return scores


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    if '--submit' in sys.argv:
        run_submit()
    else:
        run_val()
