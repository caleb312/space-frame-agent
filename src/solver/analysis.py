"""按课件主程序流程执行确定性三维求解。"""
from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

from src.model import StructuralModel
from src.model.structural import DOFS
from .assembly import digitize, dor
from .equations import SolverError, StabilityError, ldlt, slveq
from .recovery import balance, dofanli, doh
from .member_loads import assemble_member_loads, recover_sections


@dataclass
class AnalysisResult:
    model_hash: str
    units: str
    arrays: object
    K: np.ndarray
    r: np.ndarray
    dp: np.ndarray
    factors: object
    elements: list
    u: np.ndarray
    reactions: np.ndarray
    h: np.ndarray
    free_residual: np.ndarray
    force_balance: np.ndarray
    moment_balance: np.ndarray
    backward_error: float
    member_equivalent: np.ndarray | None = None
    sections: list = field(default_factory=list)
    original_member_loads: list = field(default_factory=list)

    def as_dict(self, diagnostics=False):
        force_unit = 'kN' if self.units == 'kN_m' else 'N'
        output = {
            'model_hash': self.model_hash,
            'convention': 'course_clockwise',
            'units': {'displacement': 'm', 'rotation': 'rad', 'force': force_unit, 'moment': force_unit+'*m'},
            'dof_order': list(DOFS),
            'end_force_order': ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'],
            'section_force_order': ['N', 'Qy', 'Qz', 'T', 'My', 'Mz'],
            'section_force_convention': 'textbook_section',
            'sections': self.sections,
            'member_loads': self.original_member_loads,
            'nodes': [{'id': node_id, 'displacement': self.u[6*i:6*i+6].tolist(),
                       'reaction': self.reactions[6*i:6*i+6].tolist(),
                       'restrained': self.arrays.restrained[i].tolist()}
                      for i, node_id in enumerate(self.arrays.node_ids)],
            'elements': [{'id': element_id, 'end_i': self.h[i, :6].tolist(), 'end_j': self.h[i, 6:].tolist()}
                         for i, element_id in enumerate(self.arrays.element_ids)],
            'checks': {'backward_error': self.backward_error,
                       'free_residual': self.free_residual.tolist(),
                       'force_balance': self.force_balance.tolist(),
                       'right_hand_moment_balance': self.moment_balance.tolist()},
        }
        output['units']['distributed_force'] = force_unit+'/m'
        for index, item in enumerate(output['elements']):
            item['node_i'] = self.arrays.node_ids[self.arrays.ihl[index]-1]
            item['node_j'] = self.arrays.node_ids[self.arrays.ihr[index]-1]
            item['local_basis_global'] = self.elements[index].t[:3,:3].tolist()
        if diagnostics:
            output['member_equivalent_local'] = self.member_equivalent.tolist() if self.member_equivalent is not None else []
            output['member_fixed_end_local'] = (-self.member_equivalent).tolist() if self.member_equivalent is not None else []
            output['course_arrays'] = {
                'n': self.arrays.n, 'nc': self.arrays.nc, 'm': self.arrays.m, 'nn': self.arrays.nn,
                **{name: getattr(self.arrays, name).tolist() for name in
                   ('x', 'y', 'z', 'ihl', 'ihr', 'ea', 'eiy', 'eiz', 'gj', 'eq')},
                'r': self.r.tolist(), 'dp_load': self.dp.tolist(),
                'dp_displacement': self.u[~self.arrays.restrained.ravel()].tolist(),
                'L': self.factors.L.tolist(), 'D': self.factors.D.tolist(), 'scale': self.factors.scale.tolist(),
                'K_full': self.K.tolist(), 'P_full': self.arrays.p.tolist(),
            }
            output['element_matrices'] = [
                {'id': self.arrays.element_ids[i], **{name: getattr(e, name).tolist() for name in
                 ('local', 't', 'rda', 'rdb', 'rdd', 'rde', 'ca', 'cb', 'cd', 'ce')}}
                for i, e in enumerate(self.elements)]
        return output


def solve_space_frame(model):
    """StructuralModel/dict → AnalysisResult；失败时抛出明确异常。"""
    # 再校验，避免已实例化模型被修改后绕过检查。
    data = model.model_dump(mode='json') if isinstance(model, StructuralModel) else model
    model = StructuralModel.model_validate(data)
    canonical = json.dumps(model.model_dump(mode='json'), sort_keys=True, separators=(',', ':'), allow_nan=False)
    model_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    arrays = digitize(model)                 # 数字化，变量与数组说明。
    K, r, dp, free, elements = dor(arrays)   # 单刚、对号、四分块总刚累加。
    equivalent, local_loads = assemble_member_loads(model, arrays, elements)
    dp = arrays.p[free].copy()               # 跨内荷载确定性转换后加入完整荷载与自由方程。
    try:
        factors = ldlt(r)                    # 总刚三角化，保留原 r/K。
    except StabilityError as error:
        full = int(free[error.equation])
        node_id, dof = arrays.node_ids[full//6], DOFS[full % 6]
        raise SolverError(f'{error}；消元诊断位置：节点 {node_id} / {dof}（不代表唯一问题节点）') from error
    uf = slveq(factors, dp.copy())           # 前代/回代；dp 原荷载保留。
    u = np.zeros(6 * arrays.n)
    u[free] = uf                             # 约束位移为零，回填总体位移。
    h = doh(arrays, elements, u, equivalent) # 位移项 + 固定端力，返回原荷载下真实杆端力。
    sections = recover_sections(model, arrays, h, local_loads)
    reactions, residual = dofanli(K, u, arrays.p, arrays.restrained)
    force, moment = balance(arrays, reactions)
    numerator = float(np.max(np.abs(residual[free]))) if len(free) else 0.0
    denominator = float(np.linalg.norm(r, np.inf)*np.linalg.norm(uf, np.inf)+np.linalg.norm(dp, np.inf)) if len(free) else 0.0
    backward_error = numerator / denominator if denominator else numerator
    if not all(np.all(np.isfinite(v)) for v in (h, reactions, residual, force, moment)) or not np.isfinite(backward_error):
        raise SolverError('结果恢复产生非有限数值')
    if backward_error > 1e-9:
        raise SolverError(f'方程残差超出容限：{backward_error:g}')
    return AnalysisResult(model_hash, model.units, arrays, K, r, dp, factors, elements,
                          u, reactions, h, residual[free], force, moment, backward_error,
                          equivalent, sections,
                          [load.model_dump(mode='json') for load in model.member_loads])
