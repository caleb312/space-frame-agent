"""C04 p11～13、C07 p25 及整体平衡推导。"""
import numpy as np

from .assembly import element_dofs


def doh(arrays, elements, u, equivalent=None):
    """真实局部杆端力 S=k·Tᵀ·U−p_eq；固定端力为 −p_eq。"""
    h = np.zeros((arrays.m, 12))
    for k, element in enumerate(elements, start=1):
        ue = element.t12.T @ u[element_dofs(arrays, k)]
        h[k-1] = element.local @ ue
    if equivalent is not None:
        h -= equivalent
    return h


def dofanli(K, u, p, restrained):
    """原刚度与原荷载恢复反力；自由残差单独保留。"""
    residual = K @ u - p
    reactions = np.where(restrained.ravel(), residual, 0.0)
    return reactions, residual


def balance(arrays, reactions):
    """转角/力矩是课件负右手分量：实际力矩为 X×F−M。"""
    total = (arrays.p + reactions).reshape(arrays.n, 6)
    coords = np.column_stack((arrays.x, arrays.y, arrays.z))
    force = total[:, :3].sum(axis=0)
    moment = (np.cross(coords, total[:, :3]) - total[:, 3:]).sum(axis=0)
    return force, moment
