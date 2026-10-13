"""课件 C09 四分块样式；空间补充公式见 space-frame-derivation.md。"""
from dataclasses import dataclass

import numpy as np

from .geometry import force_transform, mtmult, trnsps


def local_stiffness(length, ea, eiy, eiz, gj):
    """生成项目顺时针转角约定下的 12×12 单刚。"""
    values = np.array([length, ea, eiy, eiz, gj], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('单刚参数必须为有限正数')
    k = np.zeros((12, 12))
    # C03/C08 轴向；GJ/L 最初标准补充，现已由参考资料 p123 式5-18确认。
    pair = np.array([[1., -1.], [-1., 1.]])
    k[np.ix_([0, 6], [0, 6])] = ea / length * pair
    k[np.ix_([3, 9], [3, 9])] = gj / length * pair
    for indexes, ei, sign in (([1, 5, 7, 11], eiz, -1), ([2, 4, 8, 10], eiy, 1)):
        a, b = 12 * ei / length**3, sign * 6 * ei / length**2
        c, d = 4 * ei / length, 2 * ei / length
        k[np.ix_(indexes, indexes)] = [[a, b, -a, b], [b, c, -b, d],
                                                 [-a, -b, a, -b], [b, d, -b, c]]
    if not np.all(np.isfinite(k)):
        raise ValueError('单刚计算溢出；请检查单位及参数尺度')
    return k


@dataclass
class ElementStiffness:
    local: np.ndarray
    global_matrix: np.ndarray
    t: np.ndarray
    t12: np.ndarray
    rda: np.ndarray
    rdb: np.ndarray
    rdd: np.ndarray
    rde: np.ndarray
    ca: np.ndarray
    cb: np.ndarray
    cd: np.ndarray
    ce: np.ndarray


def stif(length, q, ea, eiy, eiz, gj):
    """C09 p25/p27：局部四分块 → 乘 tt → 乘 t → 总体四分块。"""
    local = local_stiffness(length, ea, eiy, eiz, gj)
    rda, rdb = local[:6, :6].copy(), local[:6, 6:].copy()
    rdd, rde = local[6:, :6].copy(), local[6:, 6:].copy()
    t, t12 = force_transform(q)
    tt = trnsps(t)
    aa, ab = mtmult(rda, tt), mtmult(rdb, tt)
    ad, ae = mtmult(rdd, tt), mtmult(rde, tt)
    ca, cb = mtmult(t, aa), mtmult(t, ab)
    cd, ce = mtmult(t, ad), mtmult(t, ae)
    global_matrix = np.block([[ca, cb], [cd, ce]])
    if not np.all(np.isfinite(global_matrix)):
        raise ValueError('总体单刚存在非有限数值')
    return ElementStiffness(local, global_matrix, t, t12, rda, rdb, rdd, rde, ca, cb, cd, ce)
