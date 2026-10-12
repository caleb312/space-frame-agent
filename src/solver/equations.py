"""C07 p9～11：独立 LDLT 分解、前代与回代。"""
from dataclasses import dataclass

import numpy as np


class SolverError(ValueError):
    """数值求解失败，不返回虚构位移。"""


class StabilityError(SolverError):
    def __init__(self, message, equation):
        super().__init__(message)
        self.equation = equation


@dataclass
class LDLTFactors:
    L: np.ndarray
    D: np.ndarray
    scale: np.ndarray


def ldlt(r, pivot_tolerance=1e-12):
    """对角缩放后分解；r 原阵不破坏。详见推导文档。"""
    r = np.asarray(r, dtype=float)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or not np.all(np.isfinite(r)):
        raise SolverError('ldlt: 必须为有限方阵')
    if not np.isfinite(pivot_tolerance) or pivot_tolerance <= 0:
        raise SolverError('ldlt: 主元容限必须为有限正数')
    n = r.shape[0]
    diagonal = np.diag(r)
    bad = np.flatnonzero(diagonal <= 0)
    if len(bad):
        raise StabilityError('自由方程刚度非正：可能存在无刚度自由度', int(bad[0]))
    scale = np.sqrt(diagonal)
    c = r / scale[:, None] / scale[None, :] if n else r.copy()
    if not np.allclose(c, c.T, atol=1e-10, rtol=0):
        raise SolverError('ldlt: 矩阵不对称')
    L, D = np.eye(n), np.zeros(n)
    for j in range(n):
        D[j] = c[j, j] - np.sum(L[j, :j]**2 * D[:j])
        if not np.isfinite(D[j]) or D[j] <= pivot_tolerance:
            raise StabilityError('约束后刚度奇异、非正定或数值病态；检查支座与连接', j)
        for i in range(j+1, n):
            L[i, j] = (c[i, j] - np.sum(L[i, :j] * L[j, :j] * D[:j])) / D[j]
    return LDLTFactors(L, D, scale)


def slveq(factors, dp):
    """Ld y=dp/s，z=y/D，Ldᵀ v=z，U=v/s。"""
    L, D, scale = factors.L, factors.D, factors.scale
    dp = np.asarray(dp, dtype=float)
    n = len(D)
    if dp.shape != (n,) or not np.all(np.isfinite(dp)):
        raise SolverError('slveq: 载荷维度错误或包含非有限数值')
    y = np.zeros(n)
    for i in range(n):
        y[i] = dp[i] / scale[i] - np.dot(L[i, :i], y[:i])
    z = y / D
    v = np.zeros(n)
    for i in range(n-1, -1, -1):
        v[i] = z[i] - np.dot(L[i+1:, i], v[i+1:])
    u = v / scale
    if not np.all(np.isfinite(u)):
        raise SolverError('slveq: 求解产生非有限位移')
    return u
