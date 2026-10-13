"""中期演示的自定义简单模型，参考解来自解析平衡/兼容，不冒充参考资料原题。"""
import json
from pathlib import Path
import numpy as np
from src.model import parse_model
from src.solver import solve_structure
from src.agent.runtime import FrameAgent
from src.agent.explanation import result_table,quick_result_summary
from src.visualization.native import write_native_html

ROOT=Path(__file__).resolve().parents[1]


def verify_analytical(name,output):
    """独立解析静力与兼容关系；不把Solver自身结果作为参考。"""
    EA=200000000*.01;EI=200000000*.0001;L=2;P=10
    if name=='plane_truss_triangle':
        u=[0,0,10/EA,0,5/EA,-(10*np.sqrt(2)+5)/EA]
        r=[0,5,0,5,0,0];N=[5,-5*np.sqrt(2),-5*np.sqrt(2)]
        source='节点静力平衡：N底=P/2，N斜=-P/√2；EA轴向兼容求位移'
    elif name=='plane_frame_cantilever':
        u=[0,0,0,0,-P*L**3/(3*EI),P*L**2/(2*EI)]
        r=[0,P,-P*L,0,0,0];N=[0]
        source='Euler–Bernoulli悬臂解析式：uy=-PL³/(3EI)，顺时针rz=PL²/(2EI)'
    else:
        u=[0]*9+[3/EA,-4/EA,5/EA]
        r=[-3,0,0,0,4,0,0,0,-5,0,0,0];N=[-3,4,-5]
        source='三个正交轴力杆独立EA/L弹簧及节点三向平衡；L=1m'
    actual_u=[v for n in output['nodes'] for v in n['native_displacement']]
    actual_r=[v for n in output['nodes'] for v in n['native_reaction']]
    actual_n=[e['axial_force'] for e in output['elements']]
    for actual,expected,tolerance in ((actual_u,u,1e-14),(actual_r,r,1e-10),(actual_n,N,1e-10)):
        np.testing.assert_allclose(actual,expected,atol=tolerance,rtol=1e-10)
    return {'source':source,'raw_convention':'course_clockwise',
        'reference_displacements':u,'reference_reactions':r,'reference_axial_forces':N,
        'max_displacement_error':float(np.max(np.abs(np.array(actual_u)-u))),
        'max_reaction_error':float(np.max(np.abs(np.array(actual_r)-r))),
        'max_axial_error':float(np.max(np.abs(np.array(actual_n)-N))),'passed':True}


def conditions_markdown(model):
    kind=model['structure_type']
    name={'plane_truss':'简单平面桁架（三杆全铰接）','plane_frame':'简单平面刚架（悬臂）',
          'space_truss':'简单空间桁架（三杆全铰接）'}[kind]
    dofs=['ux','uy','rz'] if kind=='plane_frame' else ['ux','uy','uz'][:len(model['nodes'][0]['coordinates'])]
    lines=['# '+name,'','这是自定义解析验证算例，不是参考资料原题。',
        '结构类型：'+kind+'。采用线弹性直接刚度法，单位kN、m，坐标方向由正坐标定义。',
        '顺时针转角/力矩采用course_clockwise。',
        '各杆材料E=200000000 kN/m²，截面A=0.01 m²。']
    if kind=='plane_frame':lines.append('平面内弯曲惯性矩I=0.0001 m⁴。节点10为固定端，节点20为自由端。')
    lines+=['','## 节点','', '| 编号 | 坐标 / m | '+' / '.join(dofs)+'约束 |','| --- | --- | --- |']
    for n in model['nodes']:
        lines.append(f"| {n['id']} | {n['coordinates']} | "+' / '.join('固定' if v else '自由' for v in n['restrained'])+' |')
    lines+=['','## 构件','', '| 编号 | i端 | j端 |','| --- | --- | --- |']
    for e in model['elements']:lines.append(f"| {e['id']} | {e['node_i']} | {e['node_j']} |")
    lines+=['','## 全部荷载','']
    for load in model['loads']:
        labels=['Fx','Fy','Mz'] if kind=='plane_frame' else ['Fx','Fy','Fz'][:len(dofs)]
        values=', '.join(f'{k}={v}' for k,v in zip(labels,load['components']))
        lines.append(f"- 节点{load['node_id']}全局荷载：{values}；力kN，力矩kN·m。")
    lines+=['- 无杆上荷载。','', '结果需求：节点位移、支座反力，以及桁架各杆轴力或刚架杆端内力。',
            '本文件仅给出计算条件，不含参考答案。']
    return '\n'.join(lines)+'\n'


def make(kind,coordinates,fixities,connections,loads):
    return {'structure_type':kind,'units':'kN_m','convention':'course_clockwise',
        'nodes':[{'id':10*(i+1),'coordinates':p,'restrained':fixities[i]} for i,p in enumerate(coordinates)],
        'materials':[{'id':1,'E':200000000.0}],
        'sections':[{'id':1,'A':.01,**({'I':.0001} if kind=='plane_frame' else {})}],
        'elements':[{'id':100+i,'node_i':10*a,'node_j':10*b,'material_id':1,'section_id':1} for i,(a,b) in enumerate(connections)],
        'loads':[{'node_id':10*n,'components':v} for n,v in loads]}


def recognition(raw):
    facts=[{'path':'/'+name,'value':value,'state':'convention' if name=='convention' else 'user_given',
            'source':'project_convention' if name=='convention' else 'user','evidence':'演示模型明确给定'} for name,value in
           [('units','kN_m'),('convention','course_clockwise')]]
    for group in ('nodes','materials','sections','elements'):
        for i,row in enumerate(raw[group]):
            for name,value in row.items():
                if value is not None:facts.append({'path':f'/{group}/{i}/{name}','value':value,'state':'user_given','source':'user','evidence':'演示模型明确给定'})
    active=[0,1,5] if raw['structure_type']=='plane_frame' else [0,1] if raw['structure_type']=='plane_truss' else [0,1,2]
    loads=[]
    for i,load in enumerate(raw['loads']):
        components=[0.]*6
        for at,val in zip(active,load['components']):components[at]=val
        loads.append({'label':f'L{i+1}','kind':'node_force','target_id':load['node_id'],'components':components,
            'units':'kN_m','position':None,'interval':None,'coordinate_system':'global','source':'user','evidence':'演示节点力','raw_description':'给定节点集中力'})
    return {'structure_type':raw['structure_type'],'classification_evidence':'演示题明确给定结构类型及铰接/刚接假定',
        'observations':['这是自定义简单演示算例，理论按项目计算规则；不是原书编号算例'],
        'facts':facts,'loads':loads,'loads_complete':True,'load_inventory_evidence':'已登记全部给定荷载','questions':[],
        'requested_results':['displacements','reactions','end_forces']}


def main():
    cases={
      'plane_truss_triangle':make('plane_truss',[[0,0],[2,0],[1,1]],[[True,True],[False,True],[False,False]],[(1,2),(1,3),(2,3)],[(3,[0,-10])]),
      'plane_frame_cantilever':make('plane_frame',[[0,0],[2,0]],[[True]*3,[False]*3],[(1,2)],[(2,[0,-10,0])]),
      'space_truss_three_bar':make('space_truss',[[1,0,0],[0,1,0],[0,0,1],[0,0,0]],[[True]*3]*3+[[False]*3],[(1,4),(2,4),(3,4)],[(4,[3,-4,5])])}
    report={}
    for name,raw in cases.items():
        model=parse_model(raw).model_dump(mode='json');agent=FrameAgent(None);draft=agent.record(recognition(model))
        if draft['status']!='awaiting_confirmation':raise ValueError(draft['blockers'])
        output=solve_structure(model).as_dict()
        (ROOT/f'examples/{name}.json').write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding='utf-8')
        (ROOT/f'examples/{name}.md').write_text(conditions_markdown(model),encoding='utf-8')
        (ROOT/f'results/{name}.json').write_text(json.dumps({'model':model,'result':output,'source':'中期演示自定义算例'},ensure_ascii=False,indent=2),encoding='utf-8')
        (ROOT/f'results/{name}.recognition.json').write_text(json.dumps({'draft':draft},ensure_ascii=False,indent=2),encoding='utf-8')
        (ROOT/f'results/{name}.summary.md').write_text(result_table(output),encoding='utf-8')
        (ROOT/f'results/{name}.answer.md').write_text(quick_result_summary(output),encoding='utf-8')
        write_native_html(model,output,ROOT/f'results/{name}.3d.html')
        report[name]={'structure_type':model['structure_type'],'model_hash':output['model_hash'],
                      'analytical_verification':verify_analytical(name,output),'checks':output['checks']}
    (ROOT/'results/midterm_examples.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared:',', '.join(cases))


if __name__=='__main__':main()
