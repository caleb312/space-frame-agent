"""输入模型与校验；力学约定见 docs/theory/space-frame-derivation.md。"""
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Positive = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
Vector3 = tuple[Number, Number, Number]
Vector6 = tuple[Number, Number, Number, Number, Number, Number]
Fixity = tuple[StrictBool, StrictBool, StrictBool, StrictBool, StrictBool, StrictBool]
DOFS = ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')


class InputRecord(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Node(InputRecord):
    id: StrictInt
    coordinates: Vector3
    restrained: Fixity = (False,) * 6


class Material(InputRecord):
    id: StrictInt
    E: Positive
    G: Positive


class Section(InputRecord):
    id: StrictInt
    A: Positive
    Iy: Positive
    Iz: Positive
    J: Positive  # Saint-Venant 扭转常数，不能默认等于 Iy+Iz。


class Element(InputRecord):
    id: StrictInt
    node_i: StrictInt
    node_j: StrictInt
    material_id: StrictInt
    section_id: StrictInt
    local_y: Vector3  # 总体系中局部 y 的参考向量；不得自动猜测。


class NodeLoad(InputRecord):
    node_id: StrictInt
    components: Vector6  # Fx,Fy,Fz,Mx,My,Mz；力矩采用课件正号。


class MemberLoad(InputRecord):
    element_id: StrictInt
    kind: Literal['point_force', 'point_moment', 'uniform_force']
    coordinate_system: Literal['local', 'global']
    components: Vector3  # 力/力矩/每单位杆长的力；力矩采用课件负右手正号。
    position: Number | None = None  # 距构件 i 端，m。
    interval: tuple[Number, Number] | None = None  # 均布荷载起止位置，m。

    @model_validator(mode='after')
    def validate_load_type(self):
        if self.kind == 'uniform_force':
            if self.position is not None or self.interval is None:
                raise ValueError('均布荷载须给出 interval，不能给 position')
        elif self.position is None or self.interval is not None:
            raise ValueError('杆上集中力/力矩须给出 position，不能给 interval')
        return self


class SectionQuery(InputRecord):
    id: StrictInt
    element_id: StrictInt
    location: Literal['position', 'midspan']
    position: Number | None = None
    side: Literal['left', 'right'] = 'right'  # 荷载不连续处的左/右极限；教材默认取右侧。

    @model_validator(mode='after')
    def validate_query(self):
        if self.location == 'midspan' and self.position is not None:
            raise ValueError('跨中请求由程序计算位置，不同时给 position')
        if self.location == 'position' and self.position is None:
            raise ValueError('指定位置请求须提供 position')
        return self


class StructuralModel(InputRecord):
    units: Literal['kN_m', 'N_m']
    convention: Literal['course_clockwise'] = 'course_clockwise'
    nodes: list[Node] = Field(min_length=2)
    materials: list[Material] = Field(min_length=1)
    sections: list[Section] = Field(min_length=1)
    elements: list[Element] = Field(min_length=1)
    loads: list[NodeLoad] = Field(default_factory=list)
    member_loads: list[MemberLoad] = Field(default_factory=list)
    section_queries: list[SectionQuery] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_structure(self):
        for name in ('nodes', 'materials', 'sections', 'elements'):
            ids = [item.id for item in getattr(self, name)]
            if len(ids) != len(set(ids)):
                raise ValueError(f'{name}: 编号重复')
        nodes = {item.id: item for item in self.nodes}
        materials = {item.id for item in self.materials}
        sections = {item.id for item in self.sections}
        for element in self.elements:
            if element.node_i not in nodes or element.node_j not in nodes:
                raise ValueError(f'单元 {element.id}: 节点不存在')
            if element.material_id not in materials or element.section_id not in sections:
                raise ValueError(f'单元 {element.id}: 材料或截面不存在')
            d = np.array(nodes[element.node_j].coordinates) - nodes[element.node_i].coordinates
            length = float(np.linalg.norm(d))
            if not np.isfinite(length) or length <= 0:
                raise ValueError(f'单元 {element.id}: 零长度或无效长度')
            ex = d / length
            hint = np.array(element.local_y)
            norm = float(np.linalg.norm(hint))
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError(f'单元 {element.id}: 局部 y 参考向量无效')
            hint = hint / norm
            projected = hint - ex * np.dot(ex, hint)
            if np.linalg.norm(projected) <= 1e-10:
                raise ValueError(f'单元 {element.id}: 局部 y 参考向量与杆轴平行')
        for load in self.loads:
            if load.node_id not in nodes:
                raise ValueError(f'荷载节点 {load.node_id} 不存在')
        elements = {element.id: element for element in self.elements}
        lengths = {eid: float(np.linalg.norm(np.array(nodes[e.node_j].coordinates)
                                              - np.array(nodes[e.node_i].coordinates)))
                   for eid, e in elements.items()}
        for load in self.member_loads:
            if load.element_id not in elements:
                raise ValueError(f'杆上荷载构件 {load.element_id} 不存在')
            length = lengths[load.element_id]
            if load.kind == 'uniform_force':
                a, b = load.interval
                if not 0 <= a < b <= length:
                    raise ValueError(f'构件 {load.element_id}: 分布荷载区间须满足 0≤a<b≤L')
            elif not 0 <= load.position <= length:
                raise ValueError(f'构件 {load.element_id}: 集中荷载位置超出杆长')
        if len({q.id for q in self.section_queries}) != len(self.section_queries):
            raise ValueError('截面查询编号重复')
        for query in self.section_queries:
            if query.element_id not in elements:
                raise ValueError(f'截面查询构件 {query.element_id} 不存在')
            if query.location == 'position' and not 0 <= query.position <= lengths[query.element_id]:
                raise ValueError(f'截面 {query.id}: 查询位置超出杆长')
        return self
