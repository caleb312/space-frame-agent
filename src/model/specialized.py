"""2014教材的原生平刚/平桁/空桁输入；不使用虚拟转动刚度。"""
from typing import Literal
import numpy as np
from pydantic import Field, StrictInt, StrictBool, model_validator
from .structural import InputRecord, Number, Positive, SectionQuery, StructuralModel


class NativeNode(InputRecord):
    id: StrictInt
    coordinates: list[Number]
    restrained: list[StrictBool]


class NativeMaterial(InputRecord):
    id: StrictInt
    E: Positive


class NativeSection(InputRecord):
    id: StrictInt
    A: Positive
    I: Positive | None = None


class NativeElement(InputRecord):
    id: StrictInt
    node_i: StrictInt
    node_j: StrictInt
    material_id: StrictInt
    section_id: StrictInt


class NativeLoad(InputRecord):
    node_id: StrictInt
    components: list[Number]


class PlaneMemberLoad(InputRecord):
    element_id: StrictInt
    kind: Literal['point_force','point_moment','uniform_force']
    coordinate_system: Literal['global','local']
    components: tuple[Number,Number,Number]  # Fx,Fy,Mz（course_clockwise）
    position: Number | None = None
    interval: tuple[Number,Number] | None = None


class SpecializedModel(InputRecord):
    structure_type: Literal['plane_truss','space_truss','plane_frame']
    units: Literal['kN_m','N_m']
    convention: Literal['course_clockwise'] = 'course_clockwise'
    nodes: list[NativeNode] = Field(min_length=2)
    materials: list[NativeMaterial] = Field(min_length=1)
    sections: list[NativeSection] = Field(min_length=1)
    elements: list[NativeElement] = Field(min_length=1)
    loads: list[NativeLoad] = Field(default_factory=list)
    member_loads: list[PlaneMemberLoad] = Field(default_factory=list)
    section_queries: list[SectionQuery] = Field(default_factory=list)

    @property
    def dimension(self):
        return 3 if self.structure_type=='space_truss' else 2

    @property
    def dof_count(self):
        return 3 if self.structure_type in ('space_truss','plane_frame') else 2

    @model_validator(mode='after')
    def check_model(self):
        for field in ('nodes','elements','materials','sections','section_queries'):
            values=[item.id for item in getattr(self,field)]
            if len(values)!=len(set(values)):
                raise ValueError(field+'编号重复')
        nodes={n.id:n for n in self.nodes};materials={m.id:m for m in self.materials};sections={s.id:s for s in self.sections}
        for n in self.nodes:
            if len(n.coordinates)!=self.dimension or len(n.restrained)!=self.dof_count:
                raise ValueError(f'节点{n.id}坐标/约束维度不符合{self.structure_type}')
        lengths={}
        for e in self.elements:
            if e.node_i not in nodes or e.node_j not in nodes or e.material_id not in materials or e.section_id not in sections:
                raise ValueError(f'构件{e.id}引用不存在')
            length=float(np.linalg.norm(np.array(nodes[e.node_j].coordinates)-nodes[e.node_i].coordinates))
            if not np.isfinite(length) or length<=0:
                raise ValueError(f'构件{e.id}零长度或无效长度')
            lengths[e.id]=length
            if self.structure_type=='plane_frame' and sections[e.section_id].I is None:
                raise ValueError(f'平面刚架截面{e.section_id}缺少I')
        for load in self.loads:
            if load.node_id not in nodes or len(load.components)!=self.dof_count:
                raise ValueError('节点荷载引用或维度错误')
        if self.structure_type!='plane_frame' and self.member_loads:
            raise ValueError('桁架仅支持节点集中力，不接受杆上荷载或力矩')
        for load in self.member_loads:
            if load.element_id not in lengths:
                raise ValueError('杆上荷载目标不存在')
            L=lengths[load.element_id]
            fx,fy,m=load.components
            if load.kind=='point_moment' and (fx!=0 or fy!=0) or load.kind!='point_moment' and m!=0:
                raise ValueError('杆上力/力矩类型与分量冲突')
            if load.kind=='uniform_force':
                if load.position is not None or load.interval is None or not 0<=load.interval[0]<load.interval[1]<=L:
                    raise ValueError('均布荷载区间无效')
            elif load.interval is not None or load.position is None or not 0<=load.position<=L:
                raise ValueError('杆上集中荷载位置无效')
        for q in self.section_queries:
            if q.element_id not in lengths or q.location=='position' and not 0<=q.position<=lengths[q.element_id]:
                raise ValueError('截面目标或位置错误')
        return self


def parse_model(data):
    if hasattr(data,'model_dump'):
        data=data.model_dump(mode='json')
    if data.get('structure_type','space_frame')=='space_frame':
        return StructuralModel.model_validate({k:v for k,v in data.items() if k!='structure_type'})
    return SpecializedModel.model_validate(data)
