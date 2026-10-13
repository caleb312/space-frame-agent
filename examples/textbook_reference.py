"""参考资料原题验证的转换工具，非 StructuralModel 的跨内荷载功能。

附录IV的 RH 分量先转换为项目课件分量；跨内荷载按虚功
转换为节点荷载，保留原始信息，不能把转换后的模型说成原题只有节点荷载。
"""
import numpy as np

from src.solver.textbook_axes import textbook_basis


def hermite(x, length):
    s = x/length
    return np.array([1-3*s*s+2*s**3, length*(s-2*s*s+s**3),
                     3*s*s-2*s**3, length*(-s*s+s**3)])


def textbook_nodal_load(length, load):
    """返回附录IV右手旋转分量下的局部等效节点力；只转换登记的类型。"""
    kind, plane, q, position = load['kind'], load['plane'], load['value'], load['position']
    if plane not in (0,1) or not 0 <= position <= length+1e-10:
        raise ValueError('参考资料荷载的平面或作用位置无效')
    p = np.zeros(12)
    if kind in (1,2):
        if kind == 1:
            force = q*hermite(position,length)
        else:
            force = np.zeros(4)
            points, weights = np.polynomial.legendre.leggauss(3)
            for point, weight in zip(points,weights):
                force += q*weight*position/2*hermite((point+1)*position/2,length)
        indexes = [1,5,7,11] if plane == 0 else [2,4,8,10]
        signs = [1,1,1,1] if plane == 0 else [1,-1,1,-1]
        p[indexes] = force*signs
    elif kind == 3:
        # 附录IV：plane=0 为轴向力，plane=1 为轴向扭矩。
        indexes = [0,6] if plane == 0 else [3,9]
        p[indexes] = q*np.array([1-position/length,position/length])
    elif kind == 6:
        # 此处只接收已登记原题的端点集中弯矩，不扩充第一版 Solver。
        if abs(position-length) > 1e-10:
            raise ValueError('参考资料验证适配器只登记末端集中弯矩')
        p[10+plane] = q*(1 if plane == 1 else -1)
    else:
        raise ValueError('此参考资料验证荷载类型尚未登记')
    return p


def equivalent_model(raw):
    """参考资料数组 → 明确转换后的第一版节点荷载模型。"""
    data = {'units':'kN_m','convention':'course_clockwise',
            'nodes':[{'id':i+1,'coordinates':point,'restrained':[i<raw['fixed_nodes']]*6}
                     for i,point in enumerate(raw['coordinates'])],
            'materials':[{'id':1,'E':raw['E'],'G':raw['G']}],
            'sections':[{'id':1,'A':raw['A'],'Iy':raw['Iy'],'Iz':raw['Iz'],'J':raw['J']}],
            'elements':[],'loads':[]}
    bases, equivalent = [], np.zeros((len(raw['connections']),12))
    for k,(i,j) in enumerate(raw['connections'],start=1):
        q = textbook_basis(raw['coordinates'][i-1],raw['coordinates'][j-1],
                            raw['roll'][k-1],style='appendix4')
        bases.append(q)
        data['elements'].append({'id':k,'node_i':i,'node_j':j,'material_id':1,'section_id':1,
                                 'local_y':q[:,1].tolist()})
    for load in raw['member_loads']:
        index = load['element']-1
        i,j = raw['connections'][index]
        length = np.linalg.norm(np.array(raw['coordinates'][j-1])-raw['coordinates'][i-1])
        local = textbook_nodal_load(length,load)
        equivalent[index] += local
        q = bases[index]
        for node,part in ((i,local[:6]),(j,local[6:])):
            # 附录IV用右手旋转分量；课件模型中力矩反号。
            components = np.r_[q@part[:3],-q@part[3:]]
            data['loads'].append({'node_id':node,'components':components.tolist()})
    return data, equivalent, bases


def displacements_in_appendix_convention(result):
    values = result.u.reshape(-1,6).copy()
    values[:,3:] *= -1
    return values


def original_end_forces_right_hand(result, equivalent):
    """减去原跨内荷载的等效节点力，恢复原题真实杆端作用力。

    main.py 的等效节点模型只返回位移产生的杆端力；本函数属于算例验证适配器。
    """
    signs = np.tile([1,1,1,-1,-1,-1],2)
    return result.h*signs - equivalent


def section_forces_in_appendix_convention(raw, result, equivalent):
    """由原题真实左端力及截面左侧原荷载静力恢复指定截面。

    输出参考资料表中的 N,Qy,Qz,T,My,Mz；并非两端广义力向量。
    对集中力所在截面采用荷载右侧，即计入该点荷载。
    """
    end_forces = original_end_forces_right_hand(result,equivalent)
    values = []
    for section in raw['sections']:
        k, x = section['element'], section['position']
        left = end_forces[k-1,:6]
        F = left[:3].copy()
        M = left[3:].copy() - np.cross([x,0,0],left[:3])
        for load in raw['member_loads']:
            if load['element'] != k:
                continue
            kind,plane,q,a = load['kind'],load['plane'],load['value'],load['position']
            if kind == 2:
                extent = min(a,x)
                force = np.zeros(3)
                force[1+plane] = q*extent
                F += force
                M += np.cross([extent/2-x,0,0],force)
            elif a <= x+1e-12:
                if kind == 1:
                    force = np.zeros(3)
                    force[1+plane] = q
                    F += force
                    M += np.cross([a-x,0,0],force)
                elif kind == 3:
                    if plane == 0:
                        F[0] += q
                    else:
                        M[0] += q
                elif kind == 6:
                    M[1+plane] += q*(1 if plane == 1 else -1)
        positive_face = np.r_[-F,-M]
        values.append(positive_face*np.array([1,-1,-1,1,-1,1]))
    return np.array(values)
