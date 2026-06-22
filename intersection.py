import numpy as np


def compute_intersection(A, B, plane_z=0.0):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    # 公式 (2): 方向向量 v = B - A
    v = B - A

    # 公式 (3): 归一化 v_hat = v / ||v||
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return None  # A 和 B 重合，无法定义方向
    v_hat = v / norm

    # 公式 (6): 求参数 t
    # 参数方程 z = A_z + v_hat_3 * t，令 z = plane_z 解出 t
    # t = (plane_z - A_z) / v_hat_3
    if abs(v_hat[2]) < 1e-9:
        return None  # 射线水平，永远碰不到桌面

    t = (plane_z - A[2]) / v_hat[2]

    if t < 0:
        return None  # 交点在射线反方向

    # 公式 (7): 代回参数方程求交点
    Ix = A[0] + v_hat[0] * t
    Iy = A[1] + v_hat[1] * t

    return np.array([Ix, Iy, plane_z])