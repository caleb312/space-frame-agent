"""独立验证参考：Hermite 形函数能量积分，禁止调用 src.solver。

直接积分应变矩阵 B，采用 NumPy 求解，对照教学四分块/LDLT 实现。
此文件是验证工具，不是生产 Solver，也不是指定参考资料原题。
理论见 docs/theory/space-frame-derivation.md。
"""
import numpy as np


def energy_frame_reference(model):
    """返回按输入节点/单元顺序的 K、P、U、R、局部杆端力。"""
    node_index = {node.id: i for i, node in enumerate(model.nodes)}
    material = {item.id: item for item in model.materials}
    section = {item.id: item for item in model.sections}
    K = np.zeros((6*len(model.nodes), 6*len(model.nodes)))
    P = np.zeros(len(K))
    restrained = np.array([node.restrained for node in model.nodes]).ravel()
    quadrature, weights = np.polynomial.legendre.leggauss(3)
    records = []
    for element in model.elements:
        i, j = node_index[element.node_i], node_index[element.node_j]
        xi, xj = np.array(model.nodes[i].coordinates), np.array(model.nodes[j].coordinates)
        length = np.linalg.norm(xj-xi)
        ex = (xj-xi)/length
        y_hint = np.array(element.local_y, dtype=float)
        ey = y_hint - ex*np.dot(ex, y_hint)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        q = np.vstack((ex, ey, ez))  # 总体系位移 → 局部。
        transform = np.zeros((12, 12))
        for block in range(4):
            transform[3*block:3*block+3, 3*block:3*block+3] = q
        mat, sec = material[element.material_id], section[element.section_id]
        constitutive = np.diag([mat.E*sec.A, mat.G*sec.J, mat.E*sec.Iy, mat.E*sec.Iz])
        indexes = np.r_[np.arange(6*i, 6*i+6), np.arange(6*j, 6*j+6)]
        samples = []
        for point, weight in zip(quadrature, weights):
            s = (point+1)/2
            B = np.zeros((4, 12))
            B[0, [0, 6]] = [-1/length, 1/length]
            B[1, [3, 9]] = [-1/length, 1/length]
            # 直接对 Hermite 形函数求二阶导数，未写入单刚系数。
            curvature = np.array([(-6+12*s)/length**2, (-4+6*s)/length,
                                   (6-12*s)/length**2, (-2+6*s)/length])
            B[2, [2,4,8,10]] = curvature       # w′=ry。
            B[3, [1,5,7,11]] = curvature*[1,-1,1,-1]  # v′=−rz。
            factor = weight*length/2
            global_B = B@transform
            contribution = factor*global_B.T@constitutive@global_B
            # 按完整单元矩阵的单个分量散布，不使用生产四分块函数。
            for a, row in enumerate(indexes):
                for b, column in enumerate(indexes):
                    K[row, column] += contribution[a, b]
            samples.append((factor, B))
        records.append((indexes, transform, constitutive, samples))
    for load in model.loads:
        index = node_index[load.node_id]
        P[6*index:6*index+6] += load.components
    free = np.flatnonzero(~restrained)
    U = np.zeros(len(P))
    if len(free):
        U[free] = np.linalg.solve(K[np.ix_(free,free)], P[free])
    R = np.where(restrained, K@U-P, 0.)
    end_forces = []
    for indexes, transform, constitutive, samples in records:
        local_u = transform@U[indexes]
        S = np.zeros(12)
        for factor, B in samples:
            S += factor*B.T@constitutive@(B@local_u)
        end_forces.append(S)
    return K, P, U, R, np.array(end_forces)
