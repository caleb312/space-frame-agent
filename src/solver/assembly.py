"""数字化数组与 dor 四分块累加；C05、C09、C10。"""
from dataclasses import dataclass

import numpy as np

from src.model import StructuralModel
from .geometry import ch
from .stiffness import stif


@dataclass
class CourseArrays:
    node_ids: list[int]
    element_ids: list[int]
    n: int
    nc: int
    m: int
    nn: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    ihl: np.ndarray  # 内部节点号从 1 开始，与课件对应。
    ihr: np.ndarray
    ea: np.ndarray
    eiy: np.ndarray
    eiz: np.ndarray
    gj: np.ndarray
    local_y: np.ndarray
    restrained: np.ndarray
    eq: np.ndarray  # [n,6]：自由分量方程下标；约束分量为 -1。
    p: np.ndarray  # 完整节点荷载；跨内荷载组装时加入一致等效量，保留支座外荷载。


def digitize(model: StructuralModel):
    fixed = [node for node in model.nodes if all(node.restrained)]
    other = [node for node in model.nodes if not all(node.restrained)]
    nodes = fixed + other
    node_ids = [node.id for node in nodes]
    lookup = {node_id: index for index, node_id in enumerate(node_ids)}
    coords = np.array([node.coordinates for node in nodes])
    restrained = np.array([node.restrained for node in nodes], dtype=bool)
    eq = np.full((len(nodes), 6), -1, dtype=int)
    free = np.flatnonzero(~restrained.ravel())
    eq.ravel()[free] = np.arange(len(free))
    p = np.zeros(6 * len(nodes))
    for load in model.loads:
        index = lookup[load.node_id]
        p[6*index:6*index+6] += load.components
    materials = {item.id: item for item in model.materials}
    sections = {item.id: item for item in model.sections}
    ea, eiy, eiz, gj = [], [], [], []
    for element in model.elements:
        mat, sec = materials[element.material_id], sections[element.section_id]
        ea.append(mat.E * sec.A)
        eiy.append(mat.E * sec.Iy)
        eiz.append(mat.E * sec.Iz)
        gj.append(mat.G * sec.J)
    if not all(np.all(np.isfinite(v)) for v in (p, ea, eiy, eiz, gj)):
        raise ValueError('荷载或刚度参数累加/相乘溢出')
    return CourseArrays(node_ids, [e.id for e in model.elements], len(nodes), len(fixed),
                        len(model.elements), len(free), coords[:, 0], coords[:, 1], coords[:, 2],
                        np.array([lookup[e.node_i]+1 for e in model.elements]),
                        np.array([lookup[e.node_j]+1 for e in model.elements]),
                        np.array(ea), np.array(eiy), np.array(eiz), np.array(gj),
                        np.array([e.local_y for e in model.elements]), restrained, eq, p)


def i0j0(arrays, k):
    """k 为课件的 1 起始杆序号；返回两端 6 个自由分量对号下标。"""
    if not 1 <= k <= arrays.m:
        raise ValueError('杆序号超出范围')
    return arrays.eq[arrays.ihl[k-1]-1].copy(), arrays.eq[arrays.ihr[k-1]-1].copy()


def element_dofs(arrays, k):
    i, j = arrays.ihl[k-1]-1, arrays.ihr[k-1]-1
    return np.r_[np.arange(6*i, 6*i+6), np.arange(6*j, 6*j+6)]


def dor(arrays):
    """按杆循环，四分块对号累加到完整 K，再取课件未知量总刚 r。"""
    K = np.zeros((6 * arrays.n, 6 * arrays.n))
    r = np.zeros((arrays.nn, arrays.nn))
    elements = []
    for k in range(1, arrays.m + 1):
        index = k-1
        i, j = arrays.ihl[index]-1, arrays.ihr[index]-1
        xi = [arrays.x[i], arrays.y[i], arrays.z[i]]
        xj = [arrays.x[j], arrays.y[j], arrays.z[j]]
        clxy, q = ch(xi, xj, arrays.local_y[index])
        element = stif(clxy[0], q, arrays.ea[index], arrays.eiy[index], arrays.eiz[index], arrays.gj[index])
        # C10 p5～6：ca/cb/cd/ce 分别累加，不能把 cb 与 cd 当成相同矩阵。
        ii, jj = np.arange(6*i, 6*i+6), np.arange(6*j, 6*j+6)
        K[np.ix_(ii, ii)] += element.ca
        K[np.ix_(ii, jj)] += element.cb
        K[np.ix_(jj, ii)] += element.cd
        K[np.ix_(jj, jj)] += element.ce
        # 课件 i0j0 → dor：自由分量直接对号进入未知量总刚。
        i0, j0 = i0j0(arrays, k)
        for rows, cols, block in ((i0,i0,element.ca), (i0,j0,element.cb),
                                  (j0,i0,element.cd), (j0,j0,element.ce)):
            active_rows, active_cols = np.flatnonzero(rows >= 0), np.flatnonzero(cols >= 0)
            r[np.ix_(rows[active_rows], cols[active_cols])] += block[np.ix_(active_rows, active_cols)]
        elements.append(element)
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(r)):
        raise ValueError('整体刚度累加溢出')
    free = np.flatnonzero(~arrays.restrained.ravel())
    dp = arrays.p[free].copy()
    return K, r, dp, free, elements
