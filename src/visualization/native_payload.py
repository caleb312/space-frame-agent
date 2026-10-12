"""原生结构数据接入既有查看器；不改变Solver，不构造虚拟材料/转动刚度。"""
import hashlib
import json
from copy import deepcopy
import numpy as np
from src.model.specialized import SpecializedModel
from src.agent.explanation import textbook_presentation
from src.solver.specialized import local_basis
from src.solver.member_loads import LocalLoad,section_forces
from .frame import arrow,moment_arc,csv_text


def sample_native_member(model,e,result,coords,basis,length,count=65):
    """平刚按既有截面力积分恢复杆内曲线；桁架用两端位移线性插值。

    平刚式为postprocessing.py的N/EA、Mz/EI与课件v'=-rz子集，
    断点分段三点Gauss精确积分本项目集中/均布荷载，不引入剪切变形。
    """
    answers={n['id']:n for n in result['nodes']};ends={v['id']:v for v in result['elements']}
    u0=np.array(answers[e.node_i]['displacement']);u1=np.array(answers[e.node_j]['displacement'])
    frame=model.structure_type=='plane_frame';breaks={0.,float(length)};discontinuities=set();loads=[]
    for load in model.member_loads:
        if load.element_id!=e.id:continue
        vector=np.array([0,0,load.components[2]],float) if load.kind=='point_moment' else np.array([*load.components[:2],0.],float)
        if load.kind!='point_moment' and load.coordinate_system=='global':vector=basis.T@vector
        loads.append(LocalLoad(e.id,load.kind,vector,load.position,load.interval))
        if load.kind=='uniform_force':breaks.update(load.interval)
        else:breaks.add(load.position);discontinuities.add(load.position)
    mat=next(m for m in model.materials if m.id==e.material_id)
    sec=next(s for s in model.sections if s.id==e.section_id)
    end_i=np.array(ends[e.id]['end_i']);local0=basis.T@u0[:3]
    points,weights=np.polynomial.legendre.leggauss(3)
    def deformation(x):
        if not frame:return u0[:3]+x/length*(u1[:3]-u0[:3]),np.zeros(3)
        integral=np.zeros(3) # ∫N/EA, ∫Mz/EI, ∫(x-s)Mz/EI
        boundaries=sorted({0.,float(x),*(b for b in breaks if 0<b<x)})
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            for p,w in zip(points,weights):
                s=(a+b)/2+(b-a)*p/2;f=section_forces(end_i,loads,s,'right')
                integral+=w*(b-a)/2*np.array([f[0]/(mat.E*sec.A),f[5]/(mat.E*sec.I),(x-s)*f[5]/(mat.E*sec.I)])
        u=basis@np.array([local0[0]+integral[0],local0[1]-u0[5]*x+integral[2],0.])
        return u,np.array([0.,0.,u0[5]-integral[1]])
    d,r=deformation(length)
    error=float(np.max(abs(d-u1[:3])));rotation_error=float(np.max(abs(r-u1[3:]))) if frame else 0.
    if error>1e-8*max(1.,float(np.max(abs(u1[:3])))) or rotation_error>1e-8*max(1.,float(np.max(abs(u1[3:])))):
        raise ValueError(f'构件{e.id}杆内变形与节点结果不一致，不能绘图')
    positions=set(np.linspace(0,length,count).tolist())|breaks
    positions.update(s['position'] for s in result.get('sections',[]) if s['element_id']==e.id)
    samples=[]
    for x in sorted(positions):
        du,rotation=deformation(x)
        for side in (('left','right') if x in discontinuities else ('right',)):
            forces=section_forces(end_i,loads,x,side) if frame else np.array([ends[e.id]['axial_force'],0,0,0,0,0])
            samples.append(dict(position=float(x),side=side,coordinates=(coords[e.node_i]+basis[:,0]*x).tolist(),
                                displacement=du.tolist(),rotation_course=rotation.tolist(),forces=forces.tolist()))
    return dict(id=e.id,node_i=e.node_i,node_j=e.node_j,length=float(length),basis=basis.tolist(),samples=samples,
                endpoint_error_m=error,endpoint_error_rad=rotation_error)


def build_native_payload(data,result=None,title=None):
    model=SpecializedModel.model_validate(data.model_dump(mode='json') if hasattr(data,'model_dump') else data)
    if result is not None and hasattr(result,'as_dict'):result=result.as_dict()
    canonical=json.dumps(model.model_dump(mode='json'),sort_keys=True,separators=(',',':'),allow_nan=False)
    digest=hashlib.sha256(canonical.encode()).hexdigest()
    if result is not None and result.get('model_hash')!=digest:raise ValueError('图形结果与当前模型不匹配')
    frame=model.structure_type=='plane_frame';dimension=model.dimension;solved=result is not None
    names={'plane_frame':'平面刚架','plane_truss':'平面桁架','space_truss':'空间桁架'}
    dofs=[0,1,5] if frame else list(range(dimension))
    coords={n.id:np.r_[n.coordinates,0.] if dimension==2 else np.array(n.coordinates) for n in model.nodes}
    span=max(float(np.ptp(np.array(list(coords.values())),axis=0).max()),1e-6)
    display=textbook_presentation(result) if solved else None
    node_results={n['id']:n for n in display['nodes']} if solved else {}
    end_results={e['id']:e for e in display['elements']} if solved else {}
    nodes=[];elements=[];bases={}
    for n in model.nodes:
        fixed=[False]*6
        for i,b in zip(dofs,n.restrained):fixed[i]=b
        answer=node_results.get(n.id)
        nodes.append(dict(id=n.id,coordinates=coords[n.id].tolist(),restrained=fixed,
            displacement=answer['displacement'] if answer else None,reaction=answer['reaction'] if answer else None,
            u_norm=float(np.linalg.norm(answer['displacement'][:3])) if answer else None))
    for e in model.elements:
        vector=coords[e.node_j]-coords[e.node_i];L=float(np.linalg.norm(vector));n=vector/L
        basis=np.column_stack((n,[-n[1],n[0],0.],[0.,0.,1.])) if frame else local_basis(n)
        bases[e.id]=(L,basis)
        elements.append(dict(id=e.id,node_i=e.node_i,node_j=e.node_j,material_id=e.material_id,section_id=e.section_id,
            length=L,basis=basis.tolist(),start=coords[e.node_i].tolist(),end=coords[e.node_j].tolist(),
            end_i=end_results[e.id]['end_i'] if solved else None,end_j=end_results[e.id]['end_j'] if solved else None,
            axial_force=end_results[e.id]['axial_force'] if solved else None,
            curve=sample_native_member(model,e,result,coords,basis,L) if solved else None))
    glyphs=[];loads=[];unit='kN' if model.units=='kN_m' else 'N'
    def add_load(label,kind,location,vector,description,load_unit):
        location=np.array(location);vector=np.array(vector)
        if np.linalg.norm(vector)==0:return
        paths=moment_arc(location,vector,span*.045) if kind=='moment' else arrow(location,vector,span*.075)
        glyphs.append(dict(paths=paths,kind=kind,label=label+' · '+description,position=location.tolist()))
        loads.append(dict(label=label,kind=kind,position=location.tolist(),global_components=vector.tolist(),unit=load_unit,description=description))
    for i,load in enumerate(model.loads,1):
        vec=np.r_[load.components[:2],0.] if dimension==2 else np.array(load.components)
        add_load(f'节点荷载{i}','force',coords[load.node_id],vec,f'节点{load.node_id}集中力',unit)
        if frame:add_load(f'节点力矩{i}','moment',coords[load.node_id],[0,0,-load.components[2]],f'节点{load.node_id}力矩（右手）',unit+'·m')
    element_lookup={e.id:e for e in model.elements}
    for i,load in enumerate(model.member_loads,1):
        e=element_lookup[load.element_id];L,basis=bases[e.id]
        vector=np.array([0,0,-load.components[2]],float) if load.kind=='point_moment' else np.array([*load.components[:2],0.],float)
        if load.kind!='point_moment' and load.coordinate_system=='local':vector=basis@vector
        if load.kind=='uniform_force':
            a,b=load.interval;paths=[]
            for x in np.linspace(a,b,9):paths.extend(arrow(coords[e.node_i]+basis[:,0]*x,vector,span*.055))
            label=f'构件{e.id}均布力，区间[{a:g},{b:g}]m'
            glyphs.append(dict(paths=paths,kind='distributed',label=label,position=(coords[e.node_i]+basis[:,0]*(a+b)/2).tolist()))
            loads.append(dict(label=f'杆上荷载{i}',kind='distributed',element_id=e.id,interval=[a,b],
                global_components=vector.tolist(),unit=unit+'/m',description=label))
        else:add_load(f'杆上荷载{i}','moment' if load.kind=='point_moment' else 'force',
            coords[e.node_i]+basis[:,0]*load.position,vector,f'构件{e.id}，距i端{load.position:g}m',unit+'·m' if load.kind=='point_moment' else unit)
    sections=[]
    rows=display.get('sections',[]) if solved else [dict(id=q.id,element_id=q.element_id,position=bases[q.element_id][0]/2 if q.location=='midspan' else q.position,side=q.side,forces=None) for q in model.section_queries]
    for row in rows:
        section=deepcopy(row);e=element_lookup[row['element_id']];_,basis=bases[e.id]
        section['coordinates']=(coords[e.node_i]+basis[:,0]*row['position']).tolist();sections.append(section)
    definitions=[('u_norm','位移模','m'),('ux','位移 ux','m'),('uy','位移 uy','m')]
    if dimension==3:definitions.append(('uz','位移 uz','m'))
    if frame:definitions.append(('rz','转角 rz','rad'))
    definitions.append(('N','轴力 N',unit))
    if frame:definitions.extend([('Qy','剪力 Qy',unit),('Mz','弯矩 Mz',unit+'·m')])
    fields={}
    for key,label,field_unit in definitions:
        values=[]
        for e in elements:
            if not solved:continue
            sampled=[]
            for sample in e['curve']['samples']:
                if key=='u_norm':value=float(np.linalg.norm(sample['displacement']))
                elif key in ('ux','uy','uz'):value=sample['displacement'][['ux','uy','uz'].index(key)]
                elif key=='rz':value=-sample['rotation_course'][2]
                else:value=sample['forces'][['N','Qy','Qz','T','My','Mz'].index(key)]
                sampled.append(value);values.append(value)
            e['curve'].setdefault('fields',{})[key]=sampled
        maximum=max((abs(v) for v in values),default=0.)
        fields[key]=dict(label=label,unit=field_unit,min=0. if key=='u_norm' else -maximum,max=maximum or 1e-12,
            data_abs_max=maximum,all_zero=maximum==0,available=solved,group='force' if key in ('N','Qy','Mz') else 'displacement')
    max_curve=max((e['curve']['fields']['u_norm'][i] for e in elements if solved for i in range(len(e['curve']['samples']))),default=0.)
    exports={}
    if solved:
        labels=['ux_m','uy_m','uz_m','rx_RH_rad','ry_RH_rad','rz_RH_rad']
        exports['nodes']=csv_text(['node_id',*[labels[i] for i in dofs]],[[n['id'],*[n['displacement'][i] for i in dofs]] for n in nodes])
        labels=['Rx','Ry','Rz','Mx_RH','My_RH','Mz_RH']
        exports['reactions']=csv_text(['node_id',*[labels[i] for i in dofs]],[[n['id'],*[n['reaction'][i] for i in dofs]] for n in nodes if any(n['restrained'])])
        active=[0,1,5] if frame else [0];force_labels=['N','Qy','Qz','T','My','Mz']
        exports['sections']=csv_text(['section_id','element_id','position_m','side',*[force_labels[i] for i in active]],
            [[s['id'],s['element_id'],s['position'],s['side'],*[s['forces'][i] for i in active]] for s in sections])
    return dict(title=title or names[model.structure_type]+' · 三维分析',model_hash=digest,solved=solved,
        structure_type=model.structure_type,dimension=dimension,native_dof_indices=dofs,
        nodes=nodes,elements=elements,materials=model.model_dump(mode='json')['materials'],
        section_properties=model.model_dump(mode='json')['sections'],glyphs=glyphs,loads=loads,sections=sections,
        fields=fields,span=span,force_unit=unit,csv=exports,
        default_scale=min(1000.,max(1.,span*.1/max_curve)) if max_curve else 1.,
        max_node_displacement_m=max((n['u_norm'] for n in nodes if n['u_norm'] is not None),default=0.),
        max_sample_displacement_m=max_curve,
        deformation_note='平面刚架杆内变形按原始荷载与截面力积分恢复。' if frame else '桁架杆件采用两端平动的线性插值，不计算弯曲或转角。',
        notes='节点/支座为全局分量；内力为局部分量。只显示本结构实际自由度。'+('平面模型以Z=0嵌入查看器，不添加自由度。' if dimension==2 else ''))
