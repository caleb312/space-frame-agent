"""单独翻译图I-12；只生成现有接口输入并调用原Solver，不修改计算规则。"""
import hashlib
import json
from pathlib import Path
import numpy as np

from src.model import StructuralModel
from src.solver import solve_space_frame
from src.agent.explanation import result_table, quick_result_summary
from src.visualization import write_frame_html

ROOT = Path(__file__).resolve().parents[1]
TF_TO_KN = 9.80665  # 标准吨力；力、力矩和全部组合刚度同时转换。
E_REF = 210_000_000.0  # kN/m²，仅为组合刚度拆分基准，不推断原题真实材性。
G_REF = E_REF/2.6

COORDINATES = [
    [0,0,0],[8.5,0,0],[20.9,0,0],[0,0,5.8],[8.5,0,6.6],[8.5,0,13],[20.9,0,13],
    [0,15,0],[8.5,15,0],[20.9,15,0],[0,15,5.8],[8.5,15,6.6],[8.5,15,13],[20.9,15,13]]
# 教材HL/HR、EK（弹性性质类别），按左右表交错读取的构件编号。
ELEMENTS = [(1,4,1),(4,5,2),(2,5,3),(5,6,3),(6,7,4),(3,7,3),(4,11,2),(5,12,2),
            (6,13,4),(7,14,4),(8,11,1),(11,12,2),(9,12,3),(12,13,3),(13,14,4),(10,14,3)]
RIGIDITIES = [(416000,5460,5460,1e8),(1740000,42900,42900,1e8),
              (845000,29640,29640,1e8),(1990000,110760,110760,1e8)]


def prepare():
    nodes = []
    for nid, coordinates in enumerate(COORDINATES,1):
        fixed = nid in (2,3,9,10)
        pinned = nid in (1,8)
        nodes.append({'id':nid,'coordinates':coordinates,
            'restrained':[fixed or pinned]*3+[fixed]*3})
    sections = []
    for sid,(ea,eiy,eiz,gj) in enumerate(RIGIDITIES,1):
        sections.append({'id':sid,'A':ea*TF_TO_KN/E_REF,'Iy':eiy*TF_TO_KN/E_REF,
                         'Iz':eiz*TF_TO_KN/E_REF,'J':gj*TF_TO_KN/G_REF})
    elements = []
    for eid,(i,j,section) in enumerate(ELEMENTS,1):
        d=np.array(COORDINATES[j-1],float)-COORDINATES[i-1]
        # sin(alpha)=0的主轴取向：竖杆ey=全局Y；其他杆ey在全局XY平面且垂直杆轴。
        hint=np.array([-d[1],d[0],0.0]) if np.linalg.norm(d[:2])>1e-12 else np.array([0.0,1.0,0.0])
        hint/=np.linalg.norm(hint)
        elements.append({'id':eid,'node_i':i,'node_j':j,'material_id':1,'section_id':section,
                         'local_y':hint.tolist()})
    member_loads=[]
    for eid,force,position,moment_rh in [(4,12,3.0,7.2),(6,8,9.6,-4.8),(14,12,3.0,7.2),(16,8,9.6,-4.8)]:
        member_loads.append({'element_id':eid,'kind':'point_force','coordinate_system':'global',
                             'components':[0,0,-force*TF_TO_KN],'position':position})
        member_loads.append({'element_id':eid,'kind':'point_moment','coordinate_system':'global',
                             'components':[0,-moment_rh*TF_TO_KN,0],'position':position})
    queries=[]
    for element in elements:
        d=np.array(COORDINATES[element['node_j']-1])-COORDINATES[element['node_i']-1]
        length=float(np.linalg.norm(d))
        for position,side in [(0.0,'right'),(length,'left')]:
            queries.append({'id':len(queries)+1,'element_id':element['id'],'location':'position',
                            'position':position,'side':side})
    return StructuralModel.model_validate({'units':'kN_m','convention':'course_clockwise',
        'nodes':nodes,'materials':[{'id':1,'E':E_REF,'G':G_REF}],'sections':sections,
        'elements':elements,'loads':[],'member_loads':member_loads,'section_queries':queries}).model_dump(mode='json')


def main():
    model=prepare()
    path=ROOT/'examples/i12_simplified.json'
    path.write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding='utf-8')
    result=solve_space_frame(model).as_dict()
    metadata={'name':'I-12简化模型','calculation_basis':'现有2014课程Solver及已确认项目规则',
        'condition_source':'钟万勰附录一例题2，条件单独翻译',
        'simplifications':['忽略杆端刚臂，采用节点间全长弹性杆','忽略剪切变形',
                           '保留荷载偏心的集中力矩，④⑭荷载位置转换为节点端距离3m'],
        'reference_material':'E/G仅为保存EA/EI/GJ的归一化拆分基准，不代表原题真实材性',
        'original_answer_comparison':'不以原题含刚臂和剪切变形的数值作为简化模型验收值'}
    out=ROOT/'results/i12_simplified.json'
    out.write_text(json.dumps({'model':model,'result':result,'simplified_metadata':metadata},
                             ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'results/i12_simplified.summary.md').write_text('> 简化模型：忽略刚臂与剪切变形，非原题精确复现。\n\n'+result_table(result),encoding='utf-8')
    (ROOT/'results/i12_simplified.answer.md').write_text('简化模型：忽略刚臂与剪切变形。\n\n'+quick_result_summary(result),encoding='utf-8')
    write_frame_html(model,None,ROOT/'results/i12_simplified.model.html',title='I-12 简化模型 · 忽略刚臂与剪切变形（未显示求解结果）')
    write_frame_html(model,result,ROOT/'results/i12_simplified.3d.html',title='I-12 简化分析 · 2014课程Solver · 忽略刚臂与剪切变形')
    previous=json.loads((ROOT/'work/zhong_book/solver_before_translation.json').read_text(encoding='utf-8'))
    unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in previous.items())
    print(json.dumps({'model':str(path),'result':str(out),'solver_files_unchanged':unchanged,
        'nodes':len(model['nodes']),'elements':len(model['elements']),'load_records':len(model['member_loads']),
        'force_balance':result['checks']['force_balance'],'moment_balance':result['checks']['right_hand_moment_balance']},ensure_ascii=False))


if __name__=='__main__':
    main()
