"""已求解结果的独立解释与程序结果表；不建模、不调用 Solver。"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from src.model import StructuralModel, parse_model
from .providers import APIError


def load_saved_result(path):
    saved = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    model = parse_model(saved['model']).model_dump(mode='json')
    canonical = json.dumps(model, sort_keys=True, separators=(',', ':'), allow_nan=False)
    actual = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    result = saved['result']
    if result['model_hash'] != actual:
        raise ValueError('保存结果的模型哈希不一致，不能作为当前模型结果解释')
    return result


def textbook_presentation(result):
    """程序把节点/杆端旋转与力矩变为附录右手分量，截面正号保留。"""
    if result.get('convention') != 'course_clockwise':
        raise ValueError('结果正号约定无法确认，不能自动转换')
    display = deepcopy(result)
    display['convention'] = 'appendix4_right_hand'
    display['original_convention'] = 'course_clockwise'
    for node in display['nodes']:
        for key in ('displacement', 'reaction'):
            values = node[key]
            if len(values) != 6 or not all(type(v) in (float, int) and math.isfinite(v) for v in values):
                raise ValueError('节点结果维度或数值不合法')
            values[3:] = [-v for v in values[3:]]
        if display.get('structure_type')=='plane_frame':
            node['native_displacement'][2] *= -1
            node['native_reaction'][2] *= -1
    for element in display['elements']:
        for key in ('end_i', 'end_j'):
            element[key][3:] = [-v for v in element[key][3:]]
        if display.get('structure_type')=='plane_frame':
            element['native_end_forces'][2] *= -1
            element['native_end_forces'][5] *= -1
    # 输入荷载及平衡诊断仍属于原始约定，不混入展示结果。
    display.pop('member_loads', None)
    display.pop('checks', None)
    for key in ('course_arrays','element_matrices','member_equivalent_local','member_fixed_end_local'):
        display.pop(key,None)
    display['coordinate_notes'] = {'nodes_and_reactions':'global', 'element_end_forces':'element_local',
                                   'section_forces':'element_local_with_textbook_section_signs'}
    display['program_statistics'] = component_statistics(display)
    return display


def component_statistics(display):
    """只在相同单位的分量中统计；不比较力与力矩，也不推断受力/应力主导模式。"""
    kind=display.get('structure_type','space_frame')
    translations=[0,1] if kind in ('plane_truss','plane_frame') else [0,1,2]
    rotations=[5] if kind=='plane_frame' else [] if kind in ('plane_truss','space_truss') else [3,4,5]
    per_node=[]
    for node in display['nodes']:
        values=node['displacement']
        u=max(translations,key=lambda i:abs(values[i]))
        item={'node_id':node['id'],
                         'largest_abs_translation_component':{'component':['ux','uy','uz'][u],
                                                              'value':values[u], 'unit':'m'}}
        if rotations:
            r=max(rotations,key=lambda i:abs(values[i]))
            item['largest_abs_rotation_component']={'component':['rx','ry','rz'][r-3],
                                                    'value':values[r],'unit':'rad'}
        per_node.append(item)
    statistics={'scope':'逐节点、仅比较同单位分量；相同值时取分量顺序中的第一项，不代表唯一最大值',
                'per_node':per_node}
    reactions=[]
    for node in display['nodes']:
        if not any(node['restrained']):
            continue
        values=node['reaction']
        force=max(translations,key=lambda i:abs(values[i]))
        item={'node_id':node['id'],
                          'largest_abs_force_component':{'component':['Rx','Ry','Rz'][force],
                                                         'value':values[force], 'unit':display['units']['force']}}
        if rotations:
            moment=max(rotations,key=lambda i:abs(values[i]))
            item['largest_abs_moment_component']={'component':['Mx','My','Mz'][moment-3],
                                                  'value':values[moment],'unit':display['units']['moment']}
        reactions.append(item)
    statistics['per_support']=reactions
    if display['sections']:
        section=max(display['sections'],key=lambda s:abs(s['forces'][0]))
        statistics['largest_abs_sampled_axial_force']={
            'section_id':section['id'],'element_id':section['element_id'],'value':section['forces'][0],
            'unit':display['units']['force'],'scope':'仅已请求截面，不是全杆应力或全结构最大内力'}
    return statistics


def result_table(result):
    display = textbook_presentation(result)
    if display.get('structure_type'):
        return native_result_table(display)
    units = display['units']
    lines = ['# 空间刚架计算结果', '',
             '以下数值由 Python Solver 计算；结果表由程序直接生成。',
             '节点转角及反力矩已由程序转换为右手正号；原始 JSON 保持项目顺时针约定。',
             f"模型哈希：{result['model_hash']}", '', '## 节点位移', '',
             f"平动单位 {units['displacement']}，转角单位 {units['rotation']}。", '',
             '| 节点 | ux | uy | uz | rx | ry | rz |', '| --- | --- | --- | --- | --- | --- | --- |']
    number = lambda value: f'{value:.9g}' if value else '0'
    for node in sorted(display['nodes'], key=lambda item: item['id']):
        lines.append('| '+str(node['id'])+' | '+' | '.join(number(v) for v in node['displacement'])+' |')
    lines.extend(['', '## 支座反力', '', f"力单位 {units['force']}，力矩单位 {units['moment']}。", '',
                  '| 节点 | Rx | Ry | Rz | Mx | My | Mz |', '| --- | --- | --- | --- | --- | --- | --- |'])
    for node in sorted(display['nodes'], key=lambda item: item['id']):
        if any(node['restrained']):
            lines.append('| '+str(node['id'])+' | '+' | '.join(number(v) for v in node['reaction'])+' |')
    if display['sections']:
        lines.extend(['', '## 指定截面内力', '',
                      f"采用局部截面正号；N/Qy/Qz 单位 {units['force']}，T/My/Mz 单位 {units['moment']}。", '',
                      '| 截面 | 构件 | 距 i 端 / m | 侧 | N | Qy | Qz | T | My | Mz |',
                      '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |'])
        for section in display['sections']:
            lines.append('| '+str(section['id'])+' | '+str(section['element_id'])+' | '
                         +number(section['position'])+' | '+section['side']+' | '
                         +' | '.join(number(v) for v in section['forces'])+' |')
    lines.extend(['', '## 杆端内力', '',
                  '分量沿各杆局部坐标轴；杆端力矩采用本结果表的右手约定，不能与局部截面正号直接混用。', '',
                  '| 构件 | 端 | Fx | Fy | Fz | Mx | My | Mz |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |'])
    for element in display['elements']:
        for key, label in (('end_i','i'),('end_j','j')):
            lines.append('| '+str(element['id'])+' | '+label+' | '+' | '.join(number(v) for v in element[key])+' |')
    return '\n'.join(lines)


def explanation_presentation(result):
    """解释专用展示精度和方向；原始结果、结果表与图形继续用全精度。"""
    display = textbook_presentation(result)
    display['translation_directions'] = [
        {'node_id': node['id'], 'global': {
            axis: ('+'+axis if value > 0 else '-'+axis if value < 0 else '零位移')
            for axis, value in zip(('x','y','z'), node['displacement'][:3])}}
        for node in display['nodes']]
    def format_values(value):
        if isinstance(value, float):
            return float(f'{value:.6g}')
        if isinstance(value, list):
            return [format_values(item) for item in value]
        if isinstance(value, dict):
            return {key: format_values(item) for key, item in value.items()}
        return value
    display = format_values(display)
    display['display_precision'] = 'Python已按6位有效数字生成解释展示值；零附近小量保留科学计数法，不按阈值改成零。'
    return display


def quick_result_summary(result):
    """由已有 Solver 结果即时生成概要，无外部请求；统计仍由 Python 完成。"""
    display = textbook_presentation(result)
    stats = display['program_statistics']
    supports = [node['id'] for node in display['nodes'] if any(node['restrained'])]
    kind={'plane_truss':'平面桁架','space_truss':'空间桁架','plane_frame':'平面刚架'}.get(display.get('structure_type'),'空间刚架')
    lines = [f"已完成{kind} {len(display['nodes'])} 个节点、{len(display['elements'])} 根构件的线弹性分析。"]
    if stats['per_node']:
        node = max(stats['per_node'], key=lambda item: abs(item['largest_abs_translation_component']['value']))
        component = node['largest_abs_translation_component']
        lines.append(f"节点平动分量中绝对值最大的是节点{node['node_id']}的 {component['component']}"
                     f"，值为 {component['value']:.6g} {component['unit']}（全局坐标，非位移模）。")
    truss=result.get('structure_type') in ('plane_truss','space_truss')
    lines.append('已输出支座'+('、'.join(map(str,supports)) if supports else '（无约束节点）')+'的反力及'+('各杆轴力。' if truss else '各杆端内力。'))
    axial = stats.get('largest_abs_sampled_axial_force')
    if axial:
        state = '受拉' if axial['value'] > 0 else '受压' if axial['value'] < 0 else '轴力为零'
        lines.append(f"已请求截面中绝对值最大的轴力在构件{axial['element_id']}、截面{axial['section_id']}："
                     f"N={axial['value']:.6g} {axial['unit']}，按局部截面正号{state}；不代表全杆最大内力。")
    lines.append('完整数值见结果表和三维图；节点量为全局分量，杆端及截面内力为局部分量。')
    return '程序结果概要（直接读取 Solver 结果，无需等待 API）：\n\n'+'\n'.join(lines)


def native_result_table(display):
    """只输出真实原生自由度；桁架不显示虚构转角和弯矩列。"""
    kind=display['structure_type'];frame=kind=='plane_frame'
    names=display['native_dof_order'];number=lambda x:f'{x:.9g}' if x else '0'
    units=display['units']
    lines=['# '+{'plane_frame':'平面刚架','plane_truss':'平面桁架','space_truss':'空间桁架'}[kind]+'计算结果','',
        '数值由确定性Solver生成。位移/反力为全局分量；平刚展示转角和力矩为右手正号，原始JSON仍为course_clockwise。',
        f"单位：平动m，转角rad，力{units['force']}，力矩{units['moment']}。",
        '','## 节点位移','', '| 节点 | '+' | '.join(names)+' |','| --- | '+' | '.join(['---']*len(names))+' |']
    for n in display['nodes']:lines.append('| '+str(n['id'])+' | '+' | '.join(number(x) for x in n['native_displacement'])+' |')
    lines+=['','## 支座反力','', '| 节点 | '+' | '.join(['Rx','Ry','Mz'] if frame else ['Rx','Ry','Rz'][:len(names)])+' |',
            '| --- | '+' | '.join(['---']*len(names))+' |']
    for n in display['nodes']:
        if any(n['native_restrained']):lines.append('| '+str(n['id'])+' | '+' | '.join(number(x) for x in n['native_reaction'])+' |')
    if not frame:
        lines+=['','## 杆件轴力（受拉为正）','','| 构件 | N |','| --- | --- |']
        for e in display['elements']:lines.append(f"| {e['id']} | {number(e['axial_force'])} |")
    else:
        lines+=['','## 局部杆端力','','| 构件 | 端 | Fx | Fy | Mz（右手） |','| --- | --- | --- | --- | --- |']
        for e in display['elements']:
            for end in ('i','j'):lines.append('| '+str(e['id'])+' | '+end+' | '+' | '.join(number(e['end_'+end][i]) for i in (0,1,5))+' |')
    if display['sections']:
        indices=(0,1,5) if frame else (0,)
        names=['N','Qy','Mz'] if frame else ['N']
        lines+=['','## 指定截面内力（局部截面正号）','',
                '| 截面 | 构件 | 距i端 / m | 侧 | '+' | '.join(names)+' |',
                '| --- | --- | --- | --- | '+' | '.join(['---']*len(names))+' |']
        for q in display['sections']:lines.append('| '+str(q['id'])+' | '+str(q['element_id'])+' | '+number(q['position'])+' | '+q['side']+' | '+' | '.join(number(q['forces'][i]) for i in indices)+' |')
    lines+=['','模型哈希：'+display['model_hash']]
    return '\n'.join(lines)+'\n'


def explain_result(client, result, question=None, concise=False):
    display = explanation_presentation(result)
    # 单次文本请求：不携带图片、长识别历史，也不提供可重新建模/求解的工具。
    messages = [
        {'role':'system', 'content': '你是结构力学课程助手。结构计算已由确定性 Python 程序完成。'
         '只用给定结果简洁解释节点位移、支座反力和指定截面内力，不自行计算、换单位或补数字。'
         '给定节点转角、节点/杆端力矩已经由程序转换为右手正号；截面保持 textbook_section 正号。'
         '节点位移、转角和支座反力用全局坐标；杆端内力与截面内力均用各杆局部坐标。'
         '绝不把杆端力矩说成全局分量；全局/局部坐标与右手/截面正号是不同的概念。'
         '极值或分量大小比较只引用 program_statistics 中的程序统计；不自行排序、计算绝对值或比值。'
         '位移与转角不能直接比较，力与力矩不能直接比较；不要写 rz 与 uz 谁更明显。'
         'N 为负只能解释该截面受压，不能仅凭 N 数值断言该处或整体以压力为主、应力最大或存在屈曲。'
         '数值直接引用程序的6位有效数字展示值，不追加小数位，不自行换单位。'
         '问位移方向时，先引用translation_directions明确沿哪些全局正/负轴移动，再给相关数值。'
         '不要重新识图、建模、要求确认或调用工具。请用中文给出文字解释，并说明局部截面与全局节点的区别。'},
        {'role':'user', 'content': '请解释以下已求解结果：'+json.dumps(display,ensure_ascii=False,allow_nan=False)}]
    if concise:
        messages[0]['content'] += ('本次用于交互页面：最多3个短要点、约200字，只回答本次问题，不展开推导。不重复整张结果表，'
            '不逐项抄写全部节点和截面数值，不重复说明同一限制。符号与坐标说明集中写一句即可。')
    if question:
        messages[1]['content'] = ('用户针对当前已求解结果的问题：'+question+
            '\n只解释当前结果；若问题需要修改模型或新增数值计算，说明应转到建模入口，不能编造。'
            '\n已求解结果：'+json.dumps(display,ensure_ascii=False,allow_nan=False))
    message = client.complete(messages, None, 'explanation')
    if message.get('tool_calls'):
        raise APIError('解释阶段返回了工具调用，已拒绝执行；数值结果仍有效')
    content = message.get('content')
    if not isinstance(content,str) or not content.strip():
        raise APIError('模型没有返回文字解释；程序结果表已保留')
    return content.strip()
