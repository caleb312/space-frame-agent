"""生成无答案识读题；参考由静力/协调/单位荷载法独立计算。

自定义验证题，线弹性假定沿用2014教材项目约定；不是教材编号题。
不改求解器，也不调用模型API。
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.model import parse_model
from src.solver import solve_structure
from src.agent.explanation import result_table

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / 'examples' / 'recognition_round2'
ANSWERS = ROOT / 'results' / 'recognition_round2_reference'
E, A, I = 210000000., .005, .00008


def model(kind, coords, fixes, edges, load):
    return dict(structure_type=kind, units='kN_m', convention='course_clockwise',
        nodes=[dict(id=10*(k+1), coordinates=p, restrained=fixes[k]) for k,p in enumerate(coords)],
        materials=[dict(id=1,E=E)], sections=[dict(id=1,A=A,**({'I':I} if kind=='plane_frame' else {}))],
        elements=[dict(id=k+1,node_i=10*i,node_j=10*j,material_id=1,section_id=1) for k,(i,j) in enumerate(edges)],
        loads=[dict(node_id=10*len(coords),components=load)])


def truss_reference(raw):
    """静定杆力/支反力由节点静力求得，位移由各杆轴向伸长协调求得。

    未调用刚度组装、LDLT或Solver输出，不将自算结果充当参考。
    """
    xyz=np.asarray([n['coordinates'] for n in raw['nodes']],float)
    dim=xyz.shape[1]; fixed=np.array([n['restrained'] for n in raw['nodes']],bool).ravel()
    fixed_ids=np.flatnonzero(fixed); free=np.flatnonzero(~fixed)
    equilibrium=np.zeros((xyz.size,len(raw['elements'])+len(fixed_ids)))
    extension=np.zeros((len(raw['elements']),xyz.size)); lengths=[]
    for k,e in enumerate(raw['elements']):
        i=e['node_i']//10-1;j=e['node_j']//10-1
        vector=xyz[j]-xyz[i]; L=np.linalg.norm(vector); n=vector/L; lengths.append(L)
        equilibrium[dim*i:dim*(i+1),k]=n
        equilibrium[dim*j:dim*(j+1),k]=-n
        extension[k,dim*i:dim*(i+1)]=-n
        extension[k,dim*j:dim*(j+1)]=n
    for k,index in enumerate(fixed_ids):equilibrium[index,len(raw['elements'])+k]=1
    P=np.zeros(xyz.size);P[-dim:]=raw['loads'][0]['components']
    assert equilibrium.shape[0]==equilibrium.shape[1], '参考仅限静定例题'
    forces=np.linalg.solve(equilibrium,-P); N=forces[:len(raw['elements'])]
    r=np.zeros(xyz.size);r[fixed_ids]=forces[len(N):]
    u=np.zeros(xyz.size);u[free]=np.linalg.solve(extension[:,free],N*np.array(lengths)/(E*A))
    return u,r,N,'节点静力平衡求轴力及反力；逐杆伸长 NL/(EA) 与节点位移协调求位移'


def frame_reference(raw):
    """折形悬臂的单位荷载虚功积分；节点作用与右手转角功共轭。

    对上游各段使用 N=F·t，M=(r_load-r)×F+Mz；三点Gauss精确积掉
    本例线性矩函数的乘积。与直接刚度求解实现独立。
    """
    xyz=np.asarray([n['coordinates'] for n in raw['nodes']],float)
    force=np.array(raw['loads'][0]['components'],float);force[2]*=-1 # 原始CW -> RH
    end=xyz[-1]; u=np.zeros((len(xyz),3))
    gauss,weights=np.polynomial.legendre.leggauss(3)
    for node in range(1,len(xyz)):
        for component in range(3):
            virtual=np.eye(3)[component]; value=0.
            for segment in range(node):
                start=xyz[segment];vector=xyz[segment+1]-start
                L=np.linalg.norm(vector);t=vector/L
                for g,w in zip(gauss,weights):
                    point=start+t*L*(g+1)/2
                    actual_b=np.array([-(end-point)[1],(end-point)[0],1.])
                    virtual_b=np.array([-(xyz[node]-point)[1],(xyz[node]-point)[0],1.])
                    value+=w*L/2*((force[:2]@t)*(virtual[:2]@t)/(E*A)
                        +(actual_b@force)*(virtual_b@virtual)/(E*I))
            u[node,component]=value
    u[:,2]*=-1 # RH -> 保存CW
    r=np.zeros_like(u);r[0,:2]=-force[:2]
    r[0,2]=end[0]*force[1]-end[1]*force[0]+force[2]
    N=[force[:2]@((xyz[j]-xyz[i])/np.linalg.norm(xyz[j]-xyz[i])) for i,j in ((0,1),(1,2))]
    return u.ravel(),r.ravel(),np.array(N),'悬臂静力求N/M；单位荷载法积分 Nn/(EA)+Mm/(EI)，先右手功共轭后转CW'


def markdown(raw, name):
    kind=raw['structure_type']
    hypotheses={'plane_truss':'全部节点为理想铰接，杆件仅承受轴力；结构与荷载均在全局x-y平面。',
        'plane_frame':'两根杆在节点20刚接，考虑轴向及平面内弯曲变形；结构与荷载均在全局x-y平面。',
        'space_truss':'全部杆端为理想球铰，杆件仅承受轴力；采用三维全局x、y、z坐标。'}[kind]
    dofs=['ux','uy','rz'] if kind=='plane_frame' else ['ux','uy','uz'][:len(raw['nodes'][0]['coordinates'])]
    lines=[f'# 识读验证题 {name}','', '自定义验证题，不是教材原题。请根据结构假定判断类型，再建立模型，等待用户确认后计算。',
        '',hypotheses,'长度m，力kN，力矩kN·m。忽略自重、温度、支座沉降及剪切变形。',
        f'各杆 E={E:g} kN/m²，A={A:g} m²。']
    if kind=='plane_frame':lines.append(f'各杆平面内弯曲惯性矩 I={I:g} m⁴。图上力矩采用右手正号；提交求解器时转换为course_clockwise。')
    lines+=['','## 节点','', '| 节点 | 全局坐标 / m | '+' / '.join(dofs)+'约束 |','| --- | --- | --- |']
    for n in raw['nodes']:
        lines.append(f"| {n['id']} | {n['coordinates']} | "+' / '.join('固定' if b else '自由' for b in n['restrained'])+' |')
    lines+=['','## 构件','', '| 构件 | i端节点 | j端节点 |','| --- | --- | --- |']
    for e in raw['elements']:lines.append(f"| {e['id']} | {e['node_i']} | {e['node_j']} |")
    load=raw['loads'][0]; values=load['components']
    if kind=='plane_frame':description=f'Fx={values[0]:g} kN，Fy={values[1]:g} kN，右手Mz={-values[2]:g} kN·m（逆时针为正）。'
    else:description='，'.join(f'{axis}={v:g} kN' for axis,v in zip(['Fx','Fy','Fz'],values))+'。'
    lines+=['','## 全部荷载','', f"节点{load['node_id']}：{description}",
        '仅有上述节点荷载；无杆上荷载，无其他荷载。', '',
        '输出节点位移和支座反力；轴力杆输出各杆轴力，弯曲杆输出杆端内力。',
        '先核对结构类型、坐标、连接、约束、荷载和材料截面，再确认计算。',
        '本文件不含参考答案。']
    return '\n'.join(lines)+'\n'


def picture(raw,name):
    canvas=Image.new('RGB',(1600,1640),'white');d=ImageDraw.Draw(canvas)
    fontpath='C:/Windows/Fonts/msyh.ttc'
    fonts={n:ImageFont.truetype(fontpath,n) for n in (24,28,32,40,46)}
    black='#172d3b';red='#bc2931';blue='#245ea0'
    def text(x,y,t,size=28,color=black):d.text((x,y),t,font=fonts[size],fill=color)
    def line(p,q,color=black,width=6):d.line([tuple(p),tuple(q)],fill=color,width=width)
    def arrow(p,q,label=None):
        p=np.array(p,float);q=np.array(q,float);line(p,q,red,5)
        v=(q-p)/np.linalg.norm(q-p);normal=np.array([-v[1],v[0]])
        d.polygon([tuple(q),tuple(q-20*v+9*normal),tuple(q-20*v-9*normal)],fill=red)
        if label:text(*(q+np.array([15,-25])),label,28,red)
    def support(p,roller=False):
        x,y=p;d.polygon([(x,y+10),(x-22,y+45),(x+22,y+45)],outline=black,width=3)
        ground=y+58 if roller else y+47
        if roller:
            for offset in (-13,13):d.ellipse((x+offset-6,y+46,x+offset+6,y+58),outline=black,width=2)
        line((x-38,ground),(x+38,ground),black,3)
        for offset in range(-32,34,13):line((x+offset,ground+2),(x+offset-9,ground+13),black,2)
    text(70,35,f'识读验证题 {name}',46)
    text(70,100,'普通数字为节点编号，圆圈数字为构件编号；红色为荷载。',28)
    kind=raw['structure_type'];xyz=np.array([n['coordinates'] for n in raw['nodes']])
    if kind=='plane_truss':pts=np.array([(300+110*x,780-110*y) for x,y in xyz])
    elif kind=='plane_frame':pts=np.array([(350+150*x,740-150*y) for x,y in xyz])
    else:pts=np.array([(820+180*x-70*y,590+35*x+45*y-100*z) for x,y,z in xyz])
    for k,e in enumerate(raw['elements']):
        p=pts[e['node_i']//10-1];q=pts[e['node_j']//10-1];line(p,q)
        middle=(p+q)/2+np.array([35,30]);x,y=middle
        d.ellipse((x-22,y-22,x+22,y+22),fill='white',outline=black,width=2)
        box=d.textbbox((0,0),str(k+1),font=fonts[28]);text(x-(box[2]-box[0])/2,y-22,str(k+1))
    for k,(p,n) in enumerate(zip(pts,raw['nodes'])):
        x,y=p;d.ellipse((x-9,y-9,x+9,y+9),fill='white' if kind!='plane_frame' else black,outline=black,width=3)
        if kind=='plane_frame' and k==2:text(x+105,y+30,str(n['id']),32)
        else:text(x+14,y+12,str(n['id']),32)
        if kind=='plane_truss' and k<2:support(p,k==1)
        elif kind=='plane_frame' and k==0:
            line((x-45,y+10),(x+45,y+10),black,4)
            for off in range(-40,41,13):line((x+off,y+12),(x+off-10,y+25),black,2)
        elif kind=='space_truss' and k<3:support(p)
    end=pts[-1]
    if kind=='plane_truss':
        arrow(end+[-100,0],end,'Fx=+6 kN');arrow(end+[0,-115],end)
        text(end[0]-175,end[1]-155,'Fy=-20 kN',28,red)
        text(1100,680,'x →',28,blue);text(130,260,'y ↑',28,blue)
    elif kind=='plane_frame':
        arrow(end+[0,-115],end);text(end[0]+30,end[1]-135,'Fy=-10 kN',28,red)
        arrow(end,end+[140,0],'Fx=+5 kN')
        d.arc((end[0]-65,end[1]-65,end[0]+65,end[1]+65),start=40,end=310,fill=red,width=4)
        arrow(end+65*np.array([np.cos(np.pi/3),np.sin(np.pi/3)]),
              end+65*np.array([np.cos(2*np.pi/9),np.sin(2*np.pi/9)]))
        text(end[0]-210,end[1]+100,'右手 Mz=+6 kN·m（逆时针）',28,red)
        text(1210,720,'x →',28,blue);text(180,260,'y ↑',28,blue)
    else:
        text(930,170,'节点40集中力：',28,red)
        text(930,215,'(Fx,Fy,Fz)=(4,-6,-12) kN',28,red)
        arrow(end+[0,-100],end);text(end[0]-230,end[1]-120,'Fz=-12 kN',24,red)
        origin=np.array([260,530])
        for vec,label in (([100,20],'x'),([-50,32],'y'),([0,-110],'z')):
            q=origin+vec;line(origin,q,blue,3);v=np.array(vec)/np.linalg.norm(vec);n=np.array([-v[1],v[0]])
            d.polygon([tuple(q),tuple(q-15*v+6*n),tuple(q-15*v-6*n)],fill=blue);text(*q,label,28,blue)
        text(160,620,'示意图比例以坐标表为准',24,blue)
    d.line([(70,850),(1530,850)],fill='#ccd7de',width=2)
    # 简洁完整条件，避免图像必须依赖另一个附件。
    y=880
    if kind=='plane_truss':sentences=['所有节点理想铰接，杆件仅承受轴力；位于全局x-y平面。']
    elif kind=='plane_frame':sentences=['节点20刚接，节点10固定；考虑轴向及平面内弯曲变形。','结构与荷载均在全局x-y平面，右手Mz逆时针为正。']
    else:sentences=['所有杆端理想球铰，杆件仅承受轴力；空间坐标如下。']
    sentences+=['单位：m、kN、kN·m；忽略自重、温度、沉降和剪切变形。',f'各杆 E=210000000 kN/m²，A=0.005 m²'+('，I=0.00008 m⁴。' if kind=='plane_frame' else '。')]
    for s in sentences:text(70,y,s);y+=45
    text(70,y,'节点坐标与支座约束（固定指对应自由度约束）：',32);y+=48
    for n in raw['nodes']:
        if kind=='plane_frame':constraint='ux、uy、rz全部固定' if n['id']==10 else '全部自由'
        elif kind=='plane_truss':constraint={10:'ux、uy固定（铰支座）',20:'仅uy固定，ux自由（水平滚动支座）',30:'ux、uy自由'}[n['id']]
        else:constraint='ux、uy、uz固定，转动自由（球铰支座）' if n['id']!=40 else 'ux、uy、uz自由'
        text(85,y,f"节点{n['id']}：{n['coordinates']} m；{constraint}。",28);y+=42
    text(70,y,'构件连接（i端 → j端）： '+ '；'.join(f"{e['id']}：{e['node_i']} → {e['node_j']}" for e in raw['elements']),28);y+=47
    if kind=='plane_truss':loadtext='节点30：全局Fx=+6 kN、Fy=-20 kN。'
    elif kind=='plane_frame':loadtext='节点30：全局Fx=+5 kN、Fy=-10 kN，右手Mz=+6 kN·m。'
    else:loadtext='节点40：全局Fx=+4 kN、Fy=-6 kN、Fz=-12 kN。'
    text(70,y,loadtext,28,red);y+=45
    text(70,y,'仅有上述节点荷载，无杆上荷载。先判断结构类型并提交完整模型。');y+=45
    text(70,y,'输出：节点位移、支座反力，以及各杆轴力或刚架杆端内力。');y+=55
    text(70,y,'自定义验证题；本图没有参考答案。图形符号与尺寸以文字、坐标表为准。',24,blue)
    assert y<1600
    canvas.save(QUESTIONS/f'{name}.png')


def main():
    QUESTIONS.mkdir(parents=True,exist_ok=True);ANSWERS.mkdir(parents=True,exist_ok=True)
    cases={
        'A':model('plane_truss',[[0,0],[6,0],[3,4]],[[True,True],[False,True],[False,False]],[(1,2),(1,3),(2,3)],[6,-20]),
        'B':model('plane_frame',[[0,0],[0,3],[4,3]],[[True]*3,[False]*3,[False]*3],[(1,2),(2,3)],[5,-10,-6]),
        'C':model('space_truss',[[2,0,0],[0,2,0],[-2,-2,0],[0,0,3]],[[True]*3]*3+[[False]*3],[(1,4),(2,4),(3,4)],[4,-6,-12])}
    index={}
    for name,raw in cases.items():
        parsed=parse_model(raw);out=solve_structure(parsed).as_dict()
        u,r,N,source=frame_reference(raw) if name=='B' else truss_reference(raw)
        actual_u=np.array([n['native_displacement'] for n in out['nodes']]).ravel()
        actual_r=np.array([n['native_reaction'] for n in out['nodes']]).ravel()
        actual_N=np.array([e['axial_force'] for e in out['elements']])
        for actual,reference,atol in ((actual_u,u,1e-12),(actual_r,r,1e-8),(actual_N,N,1e-8)):
            np.testing.assert_allclose(actual,reference,rtol=1e-9,atol=atol)
        verification=dict(reference_source=source,raw_convention='course_clockwise',
            independent_displacement=u.tolist(),independent_reaction=r.tolist(),independent_axial_force=N.tolist(),
            max_displacement_error=float(np.max(abs(actual_u-u))),max_reaction_error=float(np.max(abs(actual_r-r))),
            max_axial_error=float(np.max(abs(actual_N-N))),passed=True)
        if name=='B':
            # 逐杆截取下游荷载的静力平衡，独立核对六个局部杆端分量。
            xyz=np.array([n['coordinates'] for n in raw['nodes']],float)
            F=np.array(raw['loads'][0]['components'][:2]);M=-raw['loads'][0]['components'][2]
            reference_ends=[]
            for i,j in ((0,1),(1,2)):
                t=(xyz[j]-xyz[i])/np.linalg.norm(xyz[j]-xyz[i]);normal=np.array([-t[1],t[0]])
                axial=F@t;shear=F@normal
                def cut_moment(at):
                    arm=xyz[-1]-xyz[at]
                    return arm[0]*F[1]-arm[1]*F[0]+M
                reference_ends.append([-axial,-shear,cut_moment(i),axial,shear,-cut_moment(j)])
            actual_ends=np.array([e['native_end_forces'] for e in out['elements']])
            np.testing.assert_allclose(actual_ends,reference_ends,atol=1e-8,rtol=1e-9)
            verification['independent_native_end_forces']=reference_ends
            verification['max_end_force_error']=float(np.max(abs(actual_ends-reference_ends)))
        (QUESTIONS/f'{name}.md').write_text(markdown(raw,name),encoding='utf-8')
        picture(raw,name)
        (ANSWERS/f'{name}.json').write_text(json.dumps(dict(model=parsed.model_dump(mode='json'),result=out,verification=verification),ensure_ascii=False,indent=2),encoding='utf-8')
        (ANSWERS/f'{name}.md').write_text(f'# 验证题{name}参考答案（不要上传给Agent）\n\n独立依据：{source}。\n\n'+result_table(out),encoding='utf-8')
        index[name]=dict(expected_type=raw['structure_type'],**verification)
    (ANSWERS/'verification.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    (QUESTIONS/'README.md').write_text('''# 第二组识读验证题

三个自定义题目，不是教材原题。图片与同名MD给出相同条件，均不含答案。
题目不直接给出求解器类型名，需要Agent从理想铰接/刚接与坐标维度判断。

建议先测试A，再B，再C。每题新建分析，使用“建模与纠正”。
第一轮只上传对应PNG，发送下面的指令：

> 请只根据本次图片判断结构类型并简述依据，登记节点坐标、构件连接、支座约束、材料截面和全部荷载，提交完整模型供我核对。未知写未知，等待我确认后计算。

不要同时上传MD或参考答案。若图片读取失败，保留截图，再新建分析导入同名MD对照，区分识图问题和建模/调用问题。
平面图使用全局x-y坐标；B图上的力矩为右手正号，提交时必须转为项目course_clockwise。
只通过这三题不能视为复杂教材识图验收。
''',encoding='utf-8')
    print(json.dumps(index,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
