"""验证向量化 constrained_dtw 与原始实现完全一致
用法: python opt/_verify_dtw_vec.py
"""
import sys
from pathlib import Path

import numpy as np

BASE = Path(r'D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛')
sys.path.insert(0, str(BASE / 'opt'))
from frame_dtw import constrained_dtw  # noqa: E402


def constrained_dtw_vec(dist, window):
    """Sakoe-Chiba 带向量化: 先算带内首列 j0 (left=inf), 再批量算 j0+1..j1-1
    (left 使用已更新的同行前缀)"""
    Q, E = dist.shape
    D = np.full((Q + 1, E + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, Q + 1):
        j0 = max(1, i - window)
        j1 = min(E + 1, i + window + 1)
        # 带内首列: D[i, j0-1] 恒 inf (列0 或带外)
        D[i, j0] = dist[i - 1, j0 - 1] + min(D[i - 1, j0], D[i - 1, j0 - 1])
        if j1 - j0 > 1:
            seg = slice(j0 + 1, j1)
            cost = dist[i - 1, j0:j1 - 1]
            up = D[i - 1, j0 + 1:j1]
            diag = D[i - 1, j0:j1 - 1]
            left = D[i, j0:j1 - 1]  # 已更新的同行前缀
            D[i, seg] = cost + np.minimum(np.minimum(up, diag), left)
    path_len = Q + E
    return D[Q, E] / max(path_len, 1)


def main():
    rng = np.random.RandomState(0)
    max_err = 0.0
    for trial in range(200):
        Q = rng.randint(8, 40)
        E = rng.randint(8, 40)
        dist = rng.rand(Q, E).astype(np.float64)
        w = int(rng.randint(3, 12))
        a = constrained_dtw(dist, w)
        b = constrained_dtw_vec(dist, w)
        err = abs(a - b)
        if err > max_err:
            max_err = err
        if err > 1e-9:
            print(f'MISMATCH trial={trial} Q={Q} E={E} w={w} orig={a:.10f} vec={b:.10f}')
            return
    print(f'200 组随机测试全部一致, max_err={max_err:.2e}')


if __name__ == '__main__':
    main()
