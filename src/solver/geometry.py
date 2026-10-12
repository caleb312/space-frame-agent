"""ch 子函数：课件 C08 p25 的杆长/余弦及明确标记的局部轴补充。"""
import numpy as np


def ch(xi, xj, local_y):
    """返回 clxy=[L,l,m,n]、Q=[ex ey ez]，Q 将局部分量转总体。"""
    xi, xj, hint = (np.asarray(v, dtype=float) for v in (xi, xj, local_y))
    if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in (xi, xj, hint)):
        raise ValueError('ch: 输入必须为有限三维向量')
    delta = xj - xi
    length = float(np.linalg.norm(delta))
    hint_norm = float(np.linalg.norm(hint))
    if not np.isfinite(length) or length <= 0 or not np.isfinite(hint_norm) or hint_norm <= 0:
        raise ValueError('ch: 杆长或局部轴向量无效')
    ex = delta / length
    hint = hint / hint_norm
    ey = hint - ex * np.dot(ex, hint)
    if np.linalg.norm(ey) <= 1e-10:
        raise ValueError('ch: 局部 y 参考向量与杆轴平行')
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    return np.r_[length, ex], np.column_stack((ex, ey, ez))


def trnsps(b):
    """C09 p29：转置并保留原阵。"""
    b = np.asarray(b, dtype=float)
    if b.ndim != 2:
        raise ValueError('trnsps: 必须是二维矩阵')
    return b.T.copy()


def mtmult(a, b):
    """C09 p30：矩阵乘法，显式检查维度。"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('mtmult: 矩阵维度不匹配')
    return a @ b


def force_transform(q):
    """三维补充：节点 6×6 力转换 t 与两端 12×12 力转换。"""
    q = np.asarray(q, dtype=float)
    if q.shape != (3, 3) or not np.all(np.isfinite(q)):
        raise ValueError('局部坐标基必须为有限 3×3 矩阵')
    if not np.allclose(q.T @ q, np.eye(3), atol=1e-10, rtol=0) or np.linalg.det(q) <= 0:
        raise ValueError('局部坐标基必须为右手正交基')
    t = np.zeros((6, 6))
    t[:3, :3] = q
    t[3:, 3:] = q
    t12 = np.zeros((12, 12))
    t12[:6, :6] = t
    t12[6:, 6:] = t
    return t, t12
