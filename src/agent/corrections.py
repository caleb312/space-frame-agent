"""识别记录的增量修改；Python 合并，随后重新走完整编译校验。"""
from copy import deepcopy
import json
from typing import Literal
from pydantic import Field,StrictInt
from .recognition import Record, Fact, RecognizedLoad, RecognizedSection, RecognitionReport
from src.solver.parameters import ParameterInput
from src.model.structural import Number


class LoadChange(Record):
    label: str = Field(min_length=1)
    components: list[Number | None] | None = Field(default=None,min_length=6,max_length=6)
    position: Number | None = None
    interval: list[Number | None] | None = None
    target_id: StrictInt | None = None
    coordinate_system: Literal['global','local'] | None = None
    distribution: Literal['uniform','other'] | None = None
    moment_direction: Literal['clockwise','counterclockwise'] | None = None
    moment_magnitude: Number | None = Field(default=None,gt=0)
    raw_description: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class RecognitionPatch(Record):
    structure_type: Literal['space_frame','plane_frame','plane_truss','space_truss','unknown'] | None = None
    classification_evidence: str | None = None
    base_record_hash: str = Field(min_length=64, max_length=64)
    facts: list[Fact] = Field(default_factory=list)
    remove_fact_paths: list[str] = Field(default_factory=list)
    loads: list[RecognizedLoad] = Field(default_factory=list)
    load_changes: list[LoadChange] = Field(default_factory=list)
    remove_load_labels: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    questions: list[str] | None = None
    loads_complete: bool | None = None
    load_inventory_evidence: str | None = None
    parameter_inputs: list[ParameterInput] | None = None
    section_queries: list[RecognizedSection] | None = None
    requested_results: list[str] | None = None


def normalize_patch_arguments(arguments):
    """仅兼容供应商的容器序列化差异，不修补物理值或删除未知事项。

    JSON数组字符串解码为数组；未修改的空字符串省略（不当作清空列表）。
    单个完整参数对象可包成列表，仍交给原有严格Schema逐项检查。
    数字、布尔、非空说明字符串等不转换。
    """
    if not isinstance(arguments,dict):return arguments
    result=deepcopy(arguments)
    fields=('facts','loads','load_changes','remove_fact_paths','remove_load_labels','observations',
            'questions','parameter_inputs','section_queries','requested_results')
    for key in fields:
        if key not in result:continue
        value=result[key]
        if isinstance(value,str):
            if not value.strip():
                del result[key]
                continue
            try:decoded=json.loads(value)
            except (ValueError,TypeError):continue
            if isinstance(decoded,list):result[key]=decoded
        elif key=='parameter_inputs' and isinstance(value,dict) and value:
            result[key]=[value]
    return result


def patch_tool_schema():
    """工具侧可选字段用省略表达未修改；数组直接暴露type/items，避免nullable歧义。"""
    schema=RecognitionPatch.model_json_schema()
    for key,field in schema['properties'].items():
        branches=field.get('anyOf',[])
        active=[b for b in branches if b.get('type')!='null']
        if len(active)==1 and len(branches)>1:
            field.pop('anyOf')
            field.update(active[0])
        if field.get('default','sentinel') is None:field.pop('default')
        if key not in schema.get('required',[]):
            field['description']=(field.get('description','')+' 未修改则省略。'+
                ('必须提交JSON数组，不能提交字符串。' if field.get('type')=='array' else '')).strip()
    return schema


def merge_patch(report, patch):
    """事实按 path、荷载按 label 替换；未提交的条目及来源保持原样。"""
    merged = deepcopy(report)
    updates = patch.model_dump(mode='json', exclude_none=True)
    for field, identity, removals in (
        ('facts','path',patch.remove_fact_paths), ('loads','label',patch.remove_load_labels)):
        previous = {item[identity]: item for item in merged[field]}
        additions = updates[field]
        names = [item[identity] for item in additions]
        if len(names) != len(set(names)) or set(names) & set(removals):
            raise ValueError('同一字段或荷载不可重复修改，或同时删除和替换')
        if set(removals) - set(previous):
            raise ValueError('删除目标不存在，请核对当前记录')
        for name in removals:
            del previous[name]
        for item in additions:
            previous[item[identity]] = item
        merged[field] = list(previous.values())
    changed_labels = set()
    by_label = {item['label']:item for item in merged['loads']}
    for change in patch.load_changes:
        if change.label not in by_label:
            raise ValueError('load_changes 的荷载label不存在，请使用当前记录的原label')
        if change.label in changed_labels or change.label in {item.label for item in patch.loads}:
            raise ValueError('同一荷载不能重复修改')
        changed_labels.add(change.label)
        physical=change.model_dump(mode='json',include=change.model_fields_set &
            {'components','position','interval','target_id','coordinate_system','distribution','moment_direction','moment_magnitude'})
        if not physical:raise ValueError('load_changes必须修改分量、作用位置/区间或目标等实际字段')
        if (physical.get('moment_direction') is not None and physical.get('moment_magnitude') is not None
                and 'components' not in physical):
            # 用户明确给了新的物理大小/转向，旧数字分量属于被替换的输入，重新由Python转换。
            physical['components']=None
        by_label[change.label].update(deepcopy(physical),raw_description=change.raw_description,
                                     source='user',evidence=change.evidence)
    merged['observations'].extend(updates['observations'])
    for field in ('structure_type','classification_evidence','questions','loads_complete','load_inventory_evidence','parameter_inputs',
                  'section_queries','requested_results'):
        if field in updates:
            merged[field] = updates[field]
    # 整份记录重新验证，不能把不合法补丁写成可确认模型。
    return RecognitionReport.model_validate(merged).model_dump(mode='json')
