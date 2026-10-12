"""识别记录与 Solver 模型分离；不计算截面常数或等效荷载。"""
from copy import deepcopy
import json
import math
import re
import numpy as np
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator
from src.model.structural import Number
from src.solver.parameters import ParameterInput


class Record(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, str_strip_whitespace=True)


class Fact(Record):
    path: str  # 例如 /nodes/0/coordinates；数组下标与外部节点 id 不同。
    value: Any
    state: Literal['observed', 'user_given', 'convention', 'assigned', 'uncertain', 'needs_calculation']
    source: Literal['image', 'user', 'project_convention', 'agent_label']
    evidence: str = Field(min_length=1)

    @model_validator(mode='after')
    def check_value(self):
        json.dumps(self.value, allow_nan=False)
        if self.state in ('uncertain', 'needs_calculation') and self.value is not None:
            raise ValueError('未知或待程序计算的字段必须为 null，禁止临时占位')
        if self.state not in ('uncertain', 'needs_calculation') and self.value is None:
            raise ValueError('已知字段不可为 null')
        if self.state == 'observed' and self.source != 'image':
            raise ValueError('observed 只能引用图片')
        if self.state == 'user_given' and self.source != 'user':
            raise ValueError('user_given 只能引用用户提供的信息')
        if self.state == 'convention' and (self.source != 'project_convention' or self.path != '/convention'):
            raise ValueError('项目约定只能设置 convention，不能填补材料、几何或荷载')
        if self.state == 'assigned' or self.source == 'agent_label':
            parts = self.path.split('/')[1:]
            allowed = (len(parts) == 3 and (
                parts[0] in ('materials', 'sections') and parts[2] == 'id'
                or parts[0] == 'elements' and parts[2] in ('material_id', 'section_id')))
            if (self.state != 'assigned' or self.source != 'agent_label'
                    or not allowed or type(self.value) is not int):
                raise ValueError('assigned/agent_label 仅允许分配材料、截面管理编号及对应引用，不能补物理参数')
        return self


class RecognizedLoad(Record):
    label: str = Field(min_length=1)
    kind: Literal['node_force', 'node_moment', 'member_point_force',
                  'member_point_moment', 'member_distributed', 'unknown']
    target_id: StrictInt | None
    components: list[Number | None] | None = Field(default=None,min_length=6,max_length=6)
    units: Literal['N_m', 'kN_m'] | None
    position: Number | None = None  # 距构件 i 端，m；未知不补零。
    interval: list[Number | None] | None = Field(default=None,min_length=2,max_length=2)
    raw_description: str = Field(min_length=1)
    source: Literal['image', 'user']
    evidence: str = Field(min_length=1)
    coordinate_system: Literal['local', 'global'] | None = None
    distribution: Literal['uniform', 'other'] | None = None
    moment_direction: Literal['clockwise','counterclockwise'] | None = None
    moment_magnitude: Number | None = Field(default=None,gt=0)

    @model_validator(mode='before')
    @classmethod
    def empty_location_is_unknown(cls,data):
        # 空容器没有物理位置含义。转成未知/不适用，后续按荷载类型严格校验。
        # 非空且不合法的区间仍拒绝；绝不填成[0,L]或把position补成0。
        if isinstance(data,dict):
            data=deepcopy(data)
            for key in ('position','interval'):
                value=data.get(key)
                if value==[] or isinstance(value,str) and value.strip() in ('','[]','null'):
                    data[key]=None
        return data

    @model_validator(mode='after')
    def check_location(self):
        if ((self.moment_direction is not None or self.moment_magnitude is not None)
                and self.kind not in ('node_moment','member_point_moment')):
            raise ValueError('物理力矩转向/大小仅用于集中力矩，不能用于力或分布荷载')
        if self.components is not None and len(self.components) != 6:
            raise ValueError('components 应有六项，未知项为 null')
        # 明确分类的纯力/纯力偶没有另一类分量；只规范不适用项，绝不补未知方向。
        if self.components is not None and self.kind != 'unknown':
            inactive = range(3) if self.kind in ('node_moment', 'member_point_moment') else range(3, 6)
            self.components = self.components.copy()
            for index in inactive:
                if self.components[index] is None:
                    self.components[index] = 0
        if self.interval is not None and len(self.interval) != 2:
            raise ValueError('interval 应为起止位置两项')
        if self.kind.startswith('node_') and (self.position is not None or self.interval is not None):
            raise ValueError('节点荷载不能带跨内位置或作用区间')
        if self.kind != 'member_distributed' and self.distribution is not None:
            raise ValueError('只有杆上分布荷载可声明 distribution')
        if self.kind == 'member_distributed' and self.position is not None:
            raise ValueError('分布荷载使用 interval，不能带集中作用位置')
        if self.kind in ('member_point_force', 'member_point_moment') and self.interval is not None:
            raise ValueError('杆上集中荷载使用 position，不能带分布区间')
        return self


class RecognizedSection(Record):
    id: StrictInt
    element_id: StrictInt | None
    location: Literal['position', 'midspan', 'i_end', 'j_end']
    position: Number | None = None
    side: Literal['left', 'right'] = 'right'
    source: Literal['image', 'user']
    evidence: str = Field(min_length=1)

    @model_validator(mode='after')
    def check_section_location(self):
        if self.location in ('midspan','i_end','j_end') and self.position is not None:
            raise ValueError('跨中或杆端位置由程序计算，不同时指定数值 position')
        return self


class RecognitionReport(Record):
    structure_type: Literal['space_frame','plane_frame','plane_truss','space_truss','unknown'] = 'space_frame'
    classification_evidence: str = ''
    observations: list[str]  # 原始尺寸、材料文字、图中坐标轴等，保持原单位。
    facts: list[Fact]
    loads: list[RecognizedLoad]
    loads_complete: StrictBool  # 是否已逐项盘点图/文字中全部荷载；不是是否能够求解。
    load_inventory_evidence: str = Field(min_length=1)
    questions: list[str]
    requested_results: list[Literal['displacements', 'reactions', 'end_forces', 'section_forces']]
    section_queries: list[RecognizedSection] = Field(default_factory=list)
    parameter_inputs: list[ParameterInput] = Field(default_factory=list)

    @model_validator(mode='after')
    def unique_load_labels(self):
        labels = [load.label for load in self.loads]
        if len(labels) != len(set(labels)):
            raise ValueError('荷载 label 不可重复，每个箭头/力矩单独记录')
        return self


FIELDS = {
    'nodes': {'id', 'coordinates', 'restrained'},
    'materials': {'id', 'E', 'G'},
    'sections': {'id', 'A', 'Iy', 'Iz', 'J'},
    'elements': {'id', 'node_i', 'node_j', 'material_id', 'section_id', 'local_y'},
}


def candidate_from_native_report(report,parameter_results=()):
    kind=report.structure_type
    fields={'nodes':{'id','coordinates','restrained'},'materials':{'id','E'},
            'sections':{'id','A','I'} if kind=='plane_frame' else {'id','A'},
            'elements':{'id','node_i','node_j','material_id','section_id'}}
    candidate={'structure_type':kind,'units':None,'convention':None,'nodes':[],'materials':[],
               'sections':[],'elements':[],'loads':[],'member_loads':[],'section_queries':[]}
    issues=[];seen=set()
    if kind=='unknown':issues.append('结构类型待确定：请明确刚接框架还是轴力桁架，以及平面/空间')
    if not report.classification_evidence.strip():issues.append('缺少结构类型判断依据，请说明维度与节点连接形式')
    for fact in report.facts:
        parts=fact.path.split('/')[1:]
        if fact.path in seen or not fact.path.startswith('/'):
            issues.append(f'{fact.path}：事实路径重复或无效');continue
        seen.add(fact.path)
        if len(parts)==1 and parts[0] in ('units','convention'):
            candidate[parts[0]]=fact.value
        elif len(parts)==3 and parts[0] in fields and parts[1].isdigit() and int(parts[1])<1000 and parts[2] in fields[parts[0]]:
            rows=candidate[parts[0]];index=int(parts[1])
            while len(rows)<=index:rows.append({key:None for key in sorted(fields[parts[0]])})
            rows[index][parts[2]]=deepcopy(fact.value)
        else:issues.append(f'{fact.path}：该结构类型不支持此事实字段')
        if fact.state in ('uncertain','needs_calculation') or contains_null(fact.value):
            issues.append(f'{fact.path}：'+('待 Python 工具计算' if fact.state=='needs_calculation' else '待确定'))
    computed,conflicts=apply_parameter_results(report,candidate,parameter_results)
    resolved={item['path'] for item in computed}
    issues=[item for item in issues if not (item.partition('：')[0] in resolved and
            item.endswith(('待 Python 工具计算','待确定')))]
    issues.extend(conflicts)
    for name in ('units','convention'):
        if candidate[name] is None:issues.append('/'+name+'：缺失')
    for group,required in fields.items():
        if not candidate[group]:issues.append('/'+group+'：缺失')
        for i,row in enumerate(candidate[group]):
            for name in required:
                path=f'/{group}/{i}/{name}'
                if contains_null(row[name]) and path not in seen:issues.append(f'{path}：缺失或不完整')
    if not report.loads_complete:issues.append('尚未完成荷载盘点，不能把未识别的荷载当作零')
    for index,load in enumerate(report.loads):
        v=load.components
        if kind=='plane_frame' and load.kind in ('node_moment','member_point_moment'):
            v,moment_issues,moment_computed=plane_moment_components(load,index)
            issues.extend(moment_issues);computed.extend(moment_computed)
            if moment_issues:continue
        elif load.moment_direction is not None or load.moment_magnitude is not None:
            issues.append(load.label+'：clockwise物理转向仅适用于平面刚架；空间力矩须给全局轴向分量');continue
        if load.target_id is None or contains_null(v) or load.units!=candidate['units']:
            issues.append(load.label+'：目标、分量或单位缺失/不一致');continue
        if kind=='space_truss':active=[0,1,2]
        elif kind=='plane_frame':active=[0,1,5]
        else:active=[0,1]
        if any(v[i]!=0 for i in range(6) if i not in active):
            issues.append(load.label+'：存在该结构类型不支持的平面外荷载/节点力矩');continue
        if load.kind.startswith('node_'):
            if load.coordinate_system!='global':issues.append(load.label+'：节点荷载必须明确为global');continue
            inactive=v[3:] if load.kind=='node_force' else v[:3]
            if any(value!=0 for value in inactive):
                issues.append(load.label+'：节点力/力矩类型与分量冲突');continue
            if load.kind=='node_moment' and kind!='plane_frame':
                issues.append(load.label+'：桁架不接受节点力矩');continue
            candidate['loads'].append({'node_id':load.target_id,'components':[v[i] for i in active]})
        elif load.kind.startswith('member_') and kind=='plane_frame':
            if load.coordinate_system is None:issues.append(load.label+'：杆上荷载坐标系未指定');continue
            mapping={'member_point_force':'point_force','member_point_moment':'point_moment','member_distributed':'uniform_force'}
            if load.kind not in mapping:issues.append(load.label+'：荷载类型未支持');continue
            if load.kind=='member_distributed' and load.distribution!='uniform':
                issues.append(load.label+'：只支持常值分布荷载');continue
            candidate['member_loads'].append({'element_id':load.target_id,'kind':mapping[load.kind],
                'coordinate_system':load.coordinate_system,'components':[v[0],v[1],v[5]],
                'position':load.position,'interval':load.interval})
        else:issues.append(load.label+'：荷载类型未知或超出当前结构类型范围，不能删掉或移到节点')
    for q in report.section_queries:
        row=q.model_dump(mode='json',exclude={'source','evidence'})
        if q.element_id is None:issues.append(f'截面{q.id}目标缺失');continue
        if q.location in ('i_end','j_end'):
            element=next((e for e in candidate['elements'] if e.get('id')==q.element_id),None)
            points=[next((n.get('coordinates') for n in candidate['nodes'] if n.get('id')==element.get(key)),None)
                    for key in ('node_i','node_j')] if element else [None,None]
            if any(contains_null(p) or not isinstance(p,(list,tuple)) for p in points):
                issues.append(f'截面{q.id}需完整坐标才能定位杆端');continue
            try:L=float(np.linalg.norm(np.array(points[1],float)-points[0]))
            except (ValueError,TypeError):issues.append(f'截面{q.id}坐标无效');continue
            if not math.isfinite(L) or L<=0:issues.append(f'截面{q.id}杆长无效');continue
            row.update(location='position',position=0.0 if q.location=='i_end' else L)
            computed.append({'path':f"/section_queries/{len(candidate['section_queries'])}/position",'value':row['position'],'source':'python_geometry'})
        candidate['section_queries'].append(row)
    if 'section_forces' in report.requested_results and not report.section_queries:
        issues.append('请求截面内力但未指定截面')
    issues.extend('待澄清：'+q for q in report.questions)
    return candidate,list(dict.fromkeys(issues)),computed


def plane_moment_components(load,index):
    """平面XY物理转向由Python变为课程正号，不让LLM再做正负转换。"""
    vector=load.components;issues=[];computed=[]
    if load.moment_direction is not None or load.moment_magnitude is not None:
        if load.moment_direction is None or load.moment_magnitude is None:
            return vector,[load.label+'：需要完整的力矩大小与物理转向'],computed
        value=load.moment_magnitude*(1 if load.moment_direction=='clockwise' else -1)
        if vector is not None and any(v is not None and v!=0 for v in vector[:5]):
            return vector,[load.label+'：平面力矩原始输入与额外力/平面外力矩分量冲突'],computed
        if vector is not None and vector[5] is not None and not math.isclose(vector[5],value,rel_tol=1e-12,abs_tol=0):
            return vector,[load.label+f'：物理转向与Mz分量冲突；课程顺时针为正，应为{value:g}。提交物理大小/方向时省略components，由Python转换'],computed
        vector=[0.,0.,0.,0.,0.,value]
        computed.append({'path':f'/load_inventory/{index}/components','load_label':load.label,
                         'value':vector,'source':'python_moment_sign'})
    # 旧格式同样检查明确的原始说明；不从含糊文字猜测或自动替换数值。
    text=load.raw_description
    ambiguous=any(w in text for w in ('未知','待确认','待确定','待澄清','是否','不是','无法','尚未'))
    senses={w for w in ('顺时针','逆时针') if w in text}
    if not ambiguous and len(senses)==1 and vector is not None and vector[5] is not None:
        sense=next(iter(senses));wrong=vector[5]<0 if sense=='顺时针' else vector[5]>0
        if wrong:
            issues.append(load.label+f'：说明为{sense}，但实际Mz={vector[5]:g}；平面刚架course_clockwise顺时针为正。请更新物理转向/大小或正确分量，不能按当前荷载求解')
    return vector,issues,computed


def candidate_from_report(report, parameter_results=()):
    if report.structure_type!='space_frame':
        return candidate_from_native_report(report,parameter_results)
    return candidate_from_space_report(report,parameter_results)


def candidate_from_space_report(report, parameter_results=()):
    """只从受检事实建模；loads 从独立荷载清单编译，AI 不能直接填入。"""
    candidate = {'units': None, 'convention': None,
                 'nodes': [], 'materials': [], 'sections': [], 'elements': [], 'loads': [],
                 'member_loads': [], 'section_queries': []}
    issues = []
    seen = set()
    for fact in report.facts:
        parts = fact.path.split('/')[1:]
        if not fact.path.startswith('/') or fact.path in seen:
            issues.append(f'{fact.path}：路径无效或重复')
            continue
        seen.add(fact.path)
        if len(parts) == 1 and parts[0] in ('units', 'convention'):
            candidate[parts[0]] = deepcopy(fact.value)
        elif (len(parts) == 3 and parts[0] in FIELDS and parts[1].isdigit()
              and int(parts[1]) < 1000 and parts[2] in FIELDS[parts[0]]):
            rows = candidate[parts[0]]
            index = int(parts[1])
            while len(rows) <= index:
                rows.append({key: None for key in sorted(FIELDS[parts[0]])})
            rows[index][parts[2]] = deepcopy(fact.value)
        else:
            issues.append(f'{fact.path}：不支持的事实路径；loads 必须进入独立荷载清单')
        if fact.state in ('uncertain', 'needs_calculation'):
            issues.append(f'{fact.path}：'+('待确定' if fact.state == 'uncertain' else '待 Python 工具计算'))
        elif contains_null(fact.value):
            issues.append(f'{fact.path}：仍有未知项')

    computed_fields, conflicts = apply_parameter_results(report, candidate, parameter_results)
    # 只移除被 Python 实际填入字段的缺失/待计算提示；路径错误、重复和冲突仍保留。
    resolved = {field['path'] for field in computed_fields}
    issues = [issue for issue in issues if not (
        issue.partition('：')[0] in resolved and issue.partition('：')[2] in
        ('待 Python 工具计算', '仍有未知项'))]
    issues.extend(conflicts)
    # AI 不提供 missing 也不会绕过：所有必填字段逐项检查。
    for field in ('units', 'convention'):
        if candidate[field] is None:
            issues.append(f'/{field}：缺失')
    for group, fields in FIELDS.items():
        if not candidate[group]:
            issues.append(f'/{group}：缺失')
        for i, row in enumerate(candidate[group]):
            for field in sorted(fields):
                path = f'/{group}/{i}/{field}'
                if contains_null(row[field]) and path not in seen:
                    issues.append(f'{path}：缺失或不完整')

    if not report.loads_complete:
        issues.append('尚未完成荷载盘点，不能把未识别的荷载当作零')
    for load in report.loads:
        if load.moment_direction is not None or load.moment_magnitude is not None:
            issues.append(load.label+'：顺/逆时针物理输入仅用于平面刚架；空间力矩需明确全局轴向分量')
            continue
        if load.kind.startswith('member_'):
            compile_member_load(load, candidate, issues)
            continue
        if load.kind == 'unknown':
            issues.append(f'{load.label}：荷载类型或位置尚未确定')
            continue
        if load.target_id is None or contains_null(load.components):
            issues.append(f'{load.label}：作用节点或荷载方向/分量缺失')
            continue
        if load.units is None or load.units != candidate['units']:
            issues.append(f'{load.label}：单位未知或与模型不一致；需要程序转换')
            continue
        if load.coordinate_system == 'local':
            issues.append(f'{load.label}：节点荷载必须给出全局分量')
            continue
        active = load.components[:3] if load.kind == 'node_force' else load.components[3:]
        inactive = load.components[3:] if load.kind == 'node_force' else load.components[:3]
        if any(v != 0 for v in inactive) or not any(v != 0 for v in active):
            issues.append(f'{load.label}：荷载类型与六分量不一致')
            continue
        candidate['loads'].append({'node_id': load.target_id, 'components': load.components.copy()})
    if 'section_forces' in report.requested_results and not report.section_queries:
        issues.append('需要指定截面所在构件及位置；跨中用 midspan，由程序计算位置')
    for query in report.section_queries:
        if query.element_id is None or query.location == 'position' and query.position is None:
            issues.append(f'截面 {query.id}：构件或位置待确定')
            continue
        compiled = query.model_dump(mode='json', exclude={'source','evidence'})
        if query.location in ('i_end','j_end'):
            element = next((e for e in candidate['elements'] if e.get('id')==query.element_id),None)
            if element is None:
                issues.append(f'截面 {query.id}：杆端目标构件未声明')
                continue
            points = [next((n.get('coordinates') for n in candidate['nodes'] if n.get('id')==element.get(key)),None)
                      for key in ('node_i','node_j')]
            if any(not isinstance(p,(list,tuple)) or len(p)!=3 or contains_null(p) for p in points):
                issues.append(f'截面 {query.id}：杆端位置需完整节点坐标')
                continue
            try:
                length = float(np.linalg.norm(np.asarray(points[1],dtype=float)-np.asarray(points[0],dtype=float)))
            except (ValueError,TypeError):
                issues.append(f'截面 {query.id}：节点坐标无法计算杆长')
                continue
            if not math.isfinite(length) or length<=0:
                issues.append(f'截面 {query.id}：杆长无效')
                continue
            compiled.update(location='position',position=0.0 if query.location=='i_end' else length)
            computed_fields.append({'path':f"/section_queries/{len(candidate['section_queries'])}/position",
                'value':compiled['position'],'source':'python_geometry'})
        candidate['section_queries'].append(compiled)
    issues.extend('待澄清：'+q for q in report.questions)
    return candidate, list(dict.fromkeys(issues)), computed_fields


def compile_member_load(load, candidate, issues):
    if load.target_id is None or contains_null(load.components):
        issues.append(f'{load.label}：杆上荷载构件或方向/分量缺失，原荷载已保留')
        return
    if load.units is None or load.units != candidate['units']:
        issues.append(f'{load.label}：荷载单位未知或与模型不一致')
        return
    if load.coordinate_system is None:
        issues.append(f'{load.label}：需要明确全局/局部坐标分量，不能默认荷载方向')
        return
    is_moment = load.kind == 'member_point_moment'
    active = load.components[3:] if is_moment else load.components[:3]
    inactive = load.components[:3] if is_moment else load.components[3:]
    if any(v != 0 for v in inactive) or not any(v != 0 for v in active):
        issues.append(f'{load.label}：杆上荷载类型与分量不一致')
        return
    item = {'element_id': load.target_id, 'coordinate_system': load.coordinate_system,
            'components': active.copy()}
    if load.kind == 'member_distributed':
        if load.distribution != 'uniform':
            issues.append(f'{load.label}：需明确为均布荷载；其他分布形状尚未支持')
            return
        if contains_null(load.interval):
            issues.append(f'{load.label}：分布荷载起止位置待确定，不能默认全跨')
            return
        item.update(kind='uniform_force', interval=load.interval)
    else:
        if load.position is None:
            issues.append(f'{load.label}：杆上集中荷载位置待确定')
            return
        item.update(kind='point_moment' if is_moment else 'point_force', position=load.position)
    candidate['member_loads'].append(item)


def apply_parameter_results(report, candidate, results):
    """使用运行时保存的 Python 结果，不接受 AI 自行声明的计算值。"""
    facts = {}
    for fact in report.facts:
        facts.setdefault(fact.path, fact)
    latest = {}
    conflicts = []
    for result in results:
        if result.get('formula_version')=='truss_axial_rigidity_v1' and report.structure_type not in ('plane_truss','space_truss'):
            conflicts.append('EA等效参数只适用于轴力桁架；刚架仍须明确E/A及弯曲参数')
            continue
        for group, output in (('materials', 'material'), ('sections', 'section')):
            previous=latest.get((group,result[output]['id']))
            if (previous is not None and result.get('formula_version')=='truss_axial_rigidity_v1'
                    and previous[output]!=result[output]):
                conflicts.append(f'{group}编号{result[output]["id"]}的EA等效输入冲突；不同EA请分配独立材料/截面编号')
            latest[(group, result[output]['id'])] = result
    filled = []
    groups=(('materials','material',('E','G')),('sections','section',('A','Iy','Iz','J')))
    if report.structure_type!='space_frame':
        groups=(('materials','material',('E',)),('sections','section',('A','I') if report.structure_type=='plane_frame' else ('A',)))
    for group, output, fields in groups:
        for index, row in enumerate(candidate[group]):
            if type(row['id']) is not int:
                continue
            result = latest.get((group, row['id']))
            if result is None:
                continue
            if result['model_units'] != candidate['units']:
                conflicts.append(f'{group} 编号 {row["id"]}：参数工具结果与模型单位不一致，请重新调用工具')
                continue
            for field in fields:
                path = f'/{group}/{index}/{field}'
                fact = facts.get(path)
                if (fact is not None and fact.state == 'uncertain'
                        and result.get('formula_version')!='truss_axial_rigidity_v1'):
                    continue  # 物理含义尚未确认，不能拿一次参数计算掩盖歧义。
                value = result[output]['Iz' if field=='I' else field]
                existing = row[field]
                if existing is not None and not (type(existing) in (int, float) and
                        math.isclose(existing, value, rel_tol=1e-12, abs_tol=0)):
                    conflicts.append(f'{path}：识别值与 Python 参数结果冲突；请核对原始输入，不静默覆盖')
                    continue
                row[field] = value
                filled.append({'path': path, 'value': value,
                               'source': 'python_tool', 'calculation_id': result['calculation_id']})
    return filled, conflicts


def partial_reference_issues(report, candidate):
    """即使坐标/截面缺失，也检查编号、引用和证据中的端点编号。"""
    issues = []
    ids = {}
    for group in FIELDS:
        values = [row['id'] for row in candidate[group] if row['id'] is not None]
        valid = [v for v in values if type(v) is int]
        ids[group] = set(valid)
        if len(valid) != len(values):
            issues.append(f'{group}：编号必须为整数，不能使用数组或字符串')
        if len(valid) != len(set(valid)):
            issues.append(f'{group}：编号重复')
        if group in ('nodes', 'elements') and any(row['id'] is None for row in candidate[group]):
            issues.append(f'{group}：必须提交图中节点/构件编号，不能省略 id')
    for index, element in enumerate(candidate['elements']):
        label = element['id'] if element['id'] is not None else f'待编号条目{index+1}'
        for field, group in (('node_i', 'nodes'), ('node_j', 'nodes'),
                             ('material_id', 'materials'), ('section_id', 'sections')):
            value = element[field]
            if value is None:
                continue
            if type(value) is not int:
                issues.append(f'构件 {label} 的 {field}：必须使用整数外部编号')
            elif value not in ids[group]:
                issues.append(f'构件 {label} 的 {field}={value}：未声明对应编号，不能用数组下标代替 id')
        if (element['node_i'] is not None and element['node_i'] == element['node_j']):
            issues.append(f'构件 {label}：两端节点相同')
    for fact in report.facts:
        parts = fact.path.split('/')[1:]
        if len(parts) == 3 and parts[0] == 'elements' and parts[2] in ('node_i', 'node_j') and type(fact.value) is int:
            evidence_ids = set(map(int, re.findall(r'节点\s*(-?\d+)', fact.evidence)))
            if len(evidence_ids) == 1 and fact.value not in evidence_ids:
                issues.append(f'{fact.path}：值 {fact.value} 与证据中的节点编号 {next(iter(evidence_ids))} 不一致；请修正编号，不改动结构')
    for load in report.loads:
        group = 'nodes' if load.kind.startswith('node_') else 'elements' if load.kind.startswith('member_') else None
        if group and load.target_id is not None and load.target_id not in ids[group]:
            issues.append(f'{load.label}：作用目标 {load.target_id} 未在 {group} 声明')
    for query in report.section_queries:
        if query.element_id is not None and query.element_id not in ids['elements']:
            issues.append(f'截面 {query.id}：构件 {query.element_id} 未声明')
    return list(dict.fromkeys(issues))


def contains_null(value):
    return value is None or isinstance(value, (list, dict)) and any(
        contains_null(v) for v in (value.values() if isinstance(value, dict) else value))


def readable(value):
    return '待确定' if value is None else json.dumps(value, ensure_ascii=False)


def grouped_blockers(candidate, blockers):
    """将相同类别的字段合并为人能理解的事项，详细路径仍保留在 JSON。"""
    groups = {}
    other = []
    field_names = {'coordinates': '节点坐标', 'restrained': '节点约束', 'local_y': '构件局部轴',
                   'E': 'E', 'G': 'G', 'A': 'A', 'Iy': 'Iy', 'Iz': 'Iz', 'J': 'J'}
    group_names = {'nodes': '节点', 'elements': '构件', 'materials': '材料', 'sections': '截面'}
    questions = []
    for issue in blockers:
        if issue.startswith('待澄清：'):
            questions.append(issue.removeprefix('待澄清：'))
            continue
        path, _, reason = issue.partition('：')
        parts = path.split('/')[1:]
        if (len(parts) == 3 and parts[0] in group_names and parts[1].isdigit()
                and int(parts[1]) < len(candidate[parts[0]])
                and reason in ('待确定', '待 Python 工具计算', '仍有未知项', '缺失或不完整')):
            group, index, field = parts
            row = candidate[group][int(index)]
            label = group_names[group]+str(row.get('id') if row.get('id') is not None else '?')
            category = ('待程序提供的参数计算/单位转换' if reason == '待 Python 工具计算'
                        else field_names.get(field, field)+'待核对或补充')
            detail = label+(' '+field if reason == '待 Python 工具计算' else '')
            groups.setdefault(category, []).append(detail)
        else:
            other.append(issue)
    lines = [name+'：'+ '、'.join(dict.fromkeys(items)) for name, items in groups.items()]
    lines.extend(other)
    if questions:
        lines.extend('待澄清：'+question for question in questions[:3])
        if len(questions) > 3:
            lines.append('其余澄清问题与全部字段来源可用 /model 查看。')
    return lines


def recognition_summary(report, candidate, blockers):
    """程序生成摘要，数值只展示识别记录，不做推导计算。"""
    if report.structure_type!='space_frame':
        equivalent = any(getattr(p,'parameter_type',None)=='axial_rigidity' for p in report.parameter_inputs)
        return ('【结构类型：'+report.structure_type+'】\n判断依据：'+report.classification_evidence+
                '\n节点'+str(len(candidate['nodes']))+'，构件'+str(len(candidate['elements']))+'，荷载'+str(len(report.loads))+
                ('\n仅给EA的参数由Python等效归一化，保持EA；显示E/A不代表真实材料/截面。' if equivalent else '')+
                ('\n【阻止求解】\n'+'\n'.join(blockers) if blockers else '\n基本校验通过。请核对完整模型，确认后计算。'))
    lines = ['【识别记录与模型检查】', '这些是待用户核对的识别信息，尚未求解。']
    lines.extend('观察：'+v for v in report.observations)
    for node in candidate['nodes']:
        lines.append(f"节点 {node['id']}：坐标 {readable(node['coordinates'])}，约束 {readable(node['restrained'])}")
    pressure_unit = {'kN_m': 'kN/m²', 'N_m': 'N/m²'}.get(candidate['units'], '单位待确定')
    for material in candidate['materials']:
        lines.append(f"材料 {readable(material['id'])}：E={parameter_number(material['E'])}，G={parameter_number(material['G'])} {pressure_unit}")
    for section in candidate['sections']:
        lines.append(f"截面 {readable(section['id'])}：A={parameter_number(section['A'])} m²，"
                     + '，'.join(f"{key}={parameter_number(section[key])} m⁴" for key in ('Iy', 'Iz', 'J')))
    lines.append('连接示意（仅表示连接关系，不表示实际几何）：')
    for element in candidate['elements']:
        lines.append(f"  构件 {readable(element['id'])}：节点 {readable(element['node_i'])} ── 节点 {readable(element['node_j'])}")
    labels = {'node_force': '节点力', 'node_moment': '节点力矩',
              'member_point_force': '杆上集中力', 'member_point_moment': '杆上集中力矩',
              'member_distributed': '杆上分布荷载', 'unknown': '待确定荷载'}
    for load in report.loads:
        target_type = '节点' if load.kind.startswith('node_') else '构件' if load.kind.startswith('member_') else '目标类型待确定'
        detail = '' if load.kind.startswith('node_') else (
            f'，区间 {readable(load.interval)}' if load.kind == 'member_distributed'
            else f'，位置 {readable(load.position)}')
        lines.append(f'荷载 {load.label}：{labels[load.kind]}，{target_type} {readable(load.target_id)}'
                     f'{detail}；{load.raw_description}')
    for query in report.section_queries:
        position = {'midspan':'跨中（程序按杆长计算）','i_end':'i端（程序定位）','j_end':'j端（程序按杆长定位）'}.get(query.location,readable(query.position))
        side = '右' if query.side == 'right' else '左'
        lines.append(f'截面 {query.id}：构件 {readable(query.element_id)}，位置 {position}，取{side}侧极限')
    if blockers:
        lines.append('【阻止求解的原因】')
        lines.extend(f'{i}. {v}' for i, v in enumerate(grouped_blockers(candidate, blockers), 1))
        lines.append('请补充或修正识别信息；/confirm 不会执行求解。')
    else:
        lines.append('基本校验通过。请用 /model 核对完整模型与来源，确认后输入 /confirm。')
    return '\n'.join(lines)


def parameter_number(value):
    return f'{value:.10g}' if type(value) in (int, float) else readable(value)
