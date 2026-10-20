"""Enrollment 文本发音先验模块 (无外部 G2P 依赖)

功能:
  - word_to_phoneme_features(): 提取音节数、音素长度、元音序列、辅音序列、
    首/尾音素类型、重音位置
  - phonetic_match_score(): 评分 query 声学特征与目标词发音的匹配度
  - 仅使用 enrollment 文本 (不依赖 query ASR)
  - 弱特征: duration ratio, embedding stability

用法:
  python opt/phoneme_prior.py   # val 验证
"""

import pickle
import re
from pathlib import Path

import numpy as np

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')

_VOWELS = set('aeiou')
_CONSONANTS = set('bcdfghjklmnpqrstvwxyz')


def norm(s):
    return re.sub(r"[^a-z0-9']", '', (s or '').lower())


def word_to_phoneme_features(word):
    """将单词转为音素特征向量 (不依赖 G2P 库).

    Args:
        word: 单词字符串

    Returns:
        dict with keys:
          - n_syllables:  音节数估算
          - ph_len:       字母数
          - ph_len_per_syl: 每音节字母数
          - first_is_vowel: 首字母是否元音 (0/1)
          - last_is_vowel:  尾字母是否元音 (0/1)
          - first_char:     首字母
          - last_char:      尾字母
          - vowel_seq:      元音骨架
          - cons_seq:       辅音骨架
          - stress_pos:     重音位置估算
          - word:           原词
    """
    w = word.lower().strip()
    if not w:
        return None

    # 音节数估算: 每个元音组 = 1 音节
    vowel_groups = re.findall(r'[aeiou]+', w)
    n_syllables = max(1, len(vowel_groups))

    # 首尾字母类型
    first_char = w[0]
    last_char = w[-1]

    first_is_vowel = int(first_char in _VOWELS)
    last_is_vowel = int(last_char in _VOWELS)

    # 音素长度估算
    ph_len = len(w)
    ph_len_per_syl = ph_len / n_syllables

    # 元音/辅音骨架
    vowel_seq = ''.join([c for c in w if c in _VOWELS])
    cons_seq = ''.join([c for c in w if c in _CONSONANTS])

    # 重音位置: 双音节及以上, 默认倒数第一音节
    stress_pos = -1
    if n_syllables >= 2:
        stress_pos = n_syllables - 1

    return {
        'n_syllables': n_syllables,
        'ph_len': ph_len,
        'ph_len_per_syl': ph_len_per_syl,
        'first_is_vowel': first_is_vowel,
        'last_is_vowel': last_is_vowel,
        'first_char': first_char,
        'last_char': last_char,
        'vowel_seq': vowel_seq,
        'cons_seq': cons_seq,
        'stress_pos': stress_pos,
        'word': w,
    }


def phonetic_match_score(q_feats, enroll_word):
    """基于 enrollment 文本的发音先验评分.

    不依赖 query 的 ASR 转写, 只用 query 声学特征估计.

    Args:
        q_feats: dict with 'n_frames' (int) and 'emb_stability' (float 0~1)
        enroll_word: 目标单词字符串

    Returns:
        float score in [0.01, 0.99], 越高越匹配
    """
    pf = word_to_phoneme_features(enroll_word)
    if pf is None:
        return 0.5

    score = 0.5  # neutral baseline

    # 1. 时长合理性: query 有效帧数应在合理范围
    #    每音节约 50-200ms, whisper tiny 帧率 ~50Hz
    expected_frames = pf['n_syllables'] * 12

    if 'n_frames' in q_feats:
        nf = q_feats['n_frames']
        frame_ratio = nf / max(expected_frames, 1)
        if 0.3 < frame_ratio < 3.0:
            score += 0.05
        elif frame_ratio < 0.2 or frame_ratio > 5.0:
            score -= 0.03

    # 2. 嵌入稳定性
    if 'emb_stability' in q_feats:
        score += 0.02 * (q_feats['emb_stability'] - 0.5)

    return float(np.clip(score, 0.01, 0.99))


def compute_phoneme_scores(data):
    """为数据集中所有样本计算发音先验分数.

    Args:
        data: {pid: {enroll_txt: str, ...}}

    Returns:
        dict[pid, float]
    """
    scores = {}
    for pid, it in data.items():
        word = it.get('enroll_txt', '')
        # 从数据中提取声学特征
        q_feat = {
            'n_frames': 24,  # 默认帧数
            'emb_stability': 0.5,
        }
        # 如果有帧信息, 提取有效帧数
        if 'query_frames' in it:
            qf = it['query_frames'].astype(np.float32)
            energy = np.linalg.norm(qf, axis=1)
            n_valid = int((energy > energy.max() * 0.05).sum())
            q_feat['n_frames'] = max(1, n_valid)
            # 稳定性: 有效帧的能量变异
            valid_e = energy[energy > energy.max() * 0.05]
            if len(valid_e) > 1:
                cv = float(np.std(valid_e) / (valid_e.mean() + 1e-9))
                q_feat['emb_stability'] = float(np.clip(1.0 - cv, 0.0, 1.0))

        scores[pid] = phonetic_match_score(q_feat, word)

    return scores


def roc_auc(y, s):
    y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ============================================================
# __main__: val 验证
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("发音先验 - Val 验证")
    print("=" * 60)

    vd = pickle.load(open(BASE / 'output_whisper/val_feats.pkl', 'rb'))

    for split in ['dev_seen', 'dev_unseen']:
        data = vd[split]
        pids = sorted(data.keys())
        y = np.array([data[p]['label'] for p in pids])

        print(f"\n--- {split} n={len(pids)} ---")

        # 计算发音先验分数
        ph = compute_phoneme_scores(data)
        s_ph = np.array([ph[p] for p in pids])
        auc = roc_auc(y, s_ph)
        print(f"  phonetic_prior AUC={auc:.4f}")

        # 展示几个样例的特征
        print("  样例发音特征:")
        for pid in pids[:5]:
            w = data[pid].get('enroll_txt', '')
            pf = word_to_phoneme_features(w)
            if pf:
                print(f"    {w}: syllables={pf['n_syllables']}, "
                      f"vowel_seq={pf['vowel_seq']}, "
                      f"score={ph[pid]:.3f}")

    print("\nDone!")
