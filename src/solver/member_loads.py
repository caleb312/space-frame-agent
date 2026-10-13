"""跨内荷载 → 一致节点荷载、固定端力与截面静力恢复；见 theory/member-loads.md。"""
from dataclasses import dataclass

import numpy as np
from .assembly import element_dofs


@dataclass
class LocalLoad:
    element_id: int
    kind: str
    components: np.ndarray
    position: float | None
    interval: tuple | None


def hermite(x, length):
    """三次弯曲插值 [Ni,Hi,Nj,Hj] 及对 x 的导数。"""
    s = x/length
    h = np.array([1-3*s*s+2*s**3, length*(s-2*s*s+s**3),
                  3*s*s-2*s**3, length*(-s*s+s**3)])
    derivative = np.array([(-6*s+6*s*s)/length, 1-4*s+3*s*s,
                           (6*s-6*s*s)/length, -2*s+3*s*s])
    return h, derivative


def point_vector(length, x, force=None, moment=None):
    """课件旋转下的虚功一致节点力：v'=-rz，w'=ry。"""
    if not np.isfinite(length) or length <= 0 or not np.isfinite(x) or not 0 <= x <= length:
        raise ValueError('一致节点荷载的杆长或位置无效')
    h, dh = hermite(x, length)
    linear = np.array([1-x/length, x/length])
    p = np.zeros(12)
    if force is not None:
        fx, fy, fz = force
        p[[0,6]] += fx*linear
        p[[1,5,7,11]] += fy*h*np.array([1,-1,1,-1])
        p[[2,4,8,10]] += fz*h
    if moment is not None:
        mx, my, mz = moment
        p[[3,9]] += mx*linear
        p[[2,4,8,10]] += my*dh
        p[[1,5,7,11]] += -mz*dh*np.array([1,-1,1,-1])
    if not np.all(np.isfinite(p)):
        raise ValueError('一致节点荷载计算溢出')
    return p


def consistent_vector(length, load):
    if load.kind == 'point_force':
        return point_vector(length, load.position, force=load.components)
    if load.kind == 'point_moment':
        return point_vector(length, load.position, moment=load.components)
    if load.kind != 'uniform_force':
        raise ValueError('不支持的杆上荷载类型')
    a, b = load.interval
    if not 0 <= a < b <= length:
        raise ValueError('均布荷载区间超出杆长')
    # 标准理论补充：常值 q × 三次形函数，三点 Gauss 对该积分精确。
    points, weights = np.polynomial.legendre.leggauss(3)
    p = np.zeros(12)
    for point, weight in zip(points, weights):
        x = (a+b)/2 + (b-a)*point/2
        p += weight*(b-a)/2*point_vector(length, x, force=load.components)
    if not np.all(np.isfinite(p)):
        raise ValueError('均布荷载积分溢出')
    return p


def assemble_member_loads(model, arrays, elements):
    """保留每杆局部原荷载与等效量，同时累加完整 P（含固定节点）。"""
    local = {element.id: [] for element in model.elements}
    equivalent = np.zeros((arrays.m, 12))
    lookup = {eid: i for i, eid in enumerate(arrays.element_ids)}
    for load in model.member_loads:
        index = lookup[load.element_id]
        q = elements[index].t[:3,:3]
        components = np.asarray(load.components)
        if load.coordinate_system == 'global':
            components = q.T @ components
        item = LocalLoad(load.element_id, load.kind, components.copy(), load.position, load.interval)
        local[load.element_id].append(item)
        i, j = arrays.ihl[index]-1, arrays.ihr[index]-1
        length = float(np.linalg.norm([arrays.x[j]-arrays.x[i], arrays.y[j]-arrays.y[i], arrays.z[j]-arrays.z[i]]))
        equivalent[index] += consistent_vector(length, item)
    for index, element in enumerate(elements):
        arrays.p[element_dofs(arrays, index+1)] += element.t12 @ equivalent[index]
    if not np.all(np.isfinite(arrays.p)) or not np.all(np.isfinite(equivalent)):
        raise ValueError('跨内荷载累加产生非有限数值')
    return equivalent, local


def section_forces(end_i, local_loads, position, side='right'):
    """局部截面 N/Qy/Qz/T/My/Mz，由左侧荷载的静力平衡恢复。"""
    force = end_i[:3].copy()
    moment_rh = -end_i[3:].copy() - np.cross([position,0,0], force)
    for load in local_loads:
        if load.kind == 'uniform_force':
            a, b = load.interval
            stop = min(b, position)
            if stop > a:
                f = load.components*(stop-a)
                force += f
                moment_rh += np.cross([(a+stop)/2-position,0,0], f)
        elif load.position < position or side == 'right' and load.position == position:
            if load.kind == 'point_force':
                force += load.components
                moment_rh += np.cross([load.position-position,0,0], load.components)
            else:
                moment_rh -= load.components  # 课件负右手力矩 → 物理右手分量。
    # 参考资料附录 p198～199 的截面符号，与节点/杆端广义力并非同一组正号。
    result = np.r_[-force, -moment_rh]*np.array([1,-1,-1,1,-1,1])
    if not np.all(np.isfinite(result)):
        raise ValueError('截面内力恢复产生非有限数值')
    return result


def recover_sections(model, arrays, h, local_loads):
    lookup = {eid: index for index, eid in enumerate(arrays.element_ids)}
    output = []
    for query in model.section_queries:
        index = lookup[query.element_id]
        i, j = arrays.ihl[index]-1, arrays.ihr[index]-1
        length = float(np.linalg.norm([arrays.x[j]-arrays.x[i], arrays.y[j]-arrays.y[i], arrays.z[j]-arrays.z[i]]))
        position = length/2 if query.location == 'midspan' else query.position
        values = section_forces(h[index,:6], local_loads[query.element_id], position, query.side)
        output.append({'id': query.id, 'element_id': query.element_id, 'position': position,
                       'location': query.location, 'side': query.side, 'forces': values.tolist()})
    return output
