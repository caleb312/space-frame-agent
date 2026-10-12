import numpy as np

from src.model import StructuralModel


def cantilever_data():
    return {
        'units': 'kN_m',
        'nodes': [{'id': 10, 'coordinates': [0., 0., 0.], 'restrained': [True]*6},
                  {'id': 90, 'coordinates': [2., 0., 0.]}],
        'materials': [{'id': 1, 'E': 2e8, 'G': 8e7}],
        'sections': [{'id': 1, 'A': .01, 'Iy': 2e-5, 'Iz': 4e-5, 'J': 1e-5}],
        'elements': [{'id': 100, 'node_i': 10, 'node_j': 90, 'material_id': 1,
                      'section_id': 1, 'local_y': [0., 1., 0.]}],
        'loads': [{'node_id': 90, 'components': [3., -5., 7., 2., -4., 6.]}],
    }


def cantilever_reference(load=None, length=2.):
    """独立积分轴向/弯曲/扭转微分关系得到的悬臂端柔度。"""
    fx, fy, fz, mx, my, mz = load if load is not None else [3., -5., 7., 2., -4., 6.]
    ea, eiy, eiz, gj = 2e6, 4e3, 8e3, 8e2
    l = length
    displacement = np.array([fx*l/ea, fy*l**3/(3*eiz)-mz*l*l/(2*eiz),
                             fz*l**3/(3*eiy)+my*l*l/(2*eiy), mx*l/gj,
                             fz*l*l/(2*eiy)+my*l/eiy, -fy*l*l/(2*eiz)+mz*l/eiz])
    reaction = np.array([-fx, -fy, -fz, -mx, -my-l*fz, -mz+l*fy])
    return displacement, reaction


def serial_tip_flexibility(model: StructuralModel):
    """独立力法参考：单位荷载的截面内力能量积分，不调用 Solver 函数。

    仅用于单根无分支连续悬臂链，元素按基端至尖端排列。
    C_ab=∫(NaNb/EA+TaTb/GJ+MyaMyb/EIy+MzaMzb/EIz)dx。
    """
    nodes = {n.id: np.array(n.coordinates) for n in model.nodes}
    materials = {m.id: m for m in model.materials}
    sections = {s.id: s for s in model.sections}
    tip = nodes[model.elements[-1].node_j]
    flexibility = np.zeros((6, 6))
    forces = np.column_stack((np.eye(3), np.zeros((3, 3))))
    moments_rh = np.column_stack((np.zeros((3, 3)), -np.eye(3)))
    for element in model.elements:
        start, end = nodes[element.node_i], nodes[element.node_j]
        d = end-start
        length = np.linalg.norm(d)
        ex = d/length
        ey = np.array(element.local_y) - ex*np.dot(ex, element.local_y)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        mat, sec = materials[element.material_id], sections[element.section_id]
        stiffness = np.array([mat.E*sec.A, mat.G*sec.J, mat.E*sec.Iy, mat.E*sec.Iz])
        # 两点 Gauss 对此二次积分为精确积分。
        for xi in [-1/np.sqrt(3), 1/np.sqrt(3)]:
            position = start + d*(xi+1)/2
            moment = moments_rh + np.cross((tip-position)[:, None], forces, axisa=0, axisb=0, axisc=0)
            section_forces = np.vstack((ex@forces, ex@moment, ey@moment, ez@moment))
            flexibility += length/2 * (section_forces.T / stiffness) @ section_forces
    return flexibility
