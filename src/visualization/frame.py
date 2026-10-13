"""将模型及已保存的 Solver 结果生成离线交互三维图，不请求模型 API。"""
from copy import deepcopy
import csv
import hashlib
import html
from io import StringIO
import json
from pathlib import Path
import re

import numpy as np
from src.model import StructuralModel
from src.solver.geometry import ch
from src.solver.postprocessing import sample_frame_results
from src.agent.explanation import textbook_presentation

HERE=Path(__file__).resolve().parent


def model_hash(model):
    canonical=json.dumps(model.model_dump(mode='json'),sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def arrow(tip, vector, length):
    vector=np.asarray(vector,dtype=float)
    norm=float(np.linalg.norm(vector))
    if norm==0: return []
    direction=vector/norm
    normal=np.cross(direction,np.eye(3)[int(np.argmin(np.abs(direction)))])
    normal/=np.linalg.norm(normal)
    base=tip-direction*length
    neck=tip-direction*length*.24
    return [[base.tolist(),tip.tolist()],[(neck+normal*length*.12).tolist(),tip.tolist(),
                                         (neck-normal*length*.12).tolist()]]


def moment_arc(center, vector, radius):
    vector=np.asarray(vector,dtype=float)
    norm=float(np.linalg.norm(vector))
    if norm==0: return []
    axis=vector/norm
    u=np.cross(axis,np.eye(3)[int(np.argmin(np.abs(axis)))]);u/=np.linalg.norm(u)
    v=np.cross(axis,u)
    angles=np.linspace(0,1.6*np.pi,25)
    curve=np.array([center+radius*(np.cos(a)*u+np.sin(a)*v) for a in angles])
    tangent=-np.sin(angles[-1])*u+np.cos(angles[-1])*v
    return [curve.tolist(),*arrow(curve[-1],tangent,radius*.5)[1:]]


def csv_text(header,rows):
    output=StringIO()
    writer=csv.writer(output)
    writer.writerow(header);writer.writerows(rows)
    return output.getvalue()


def build_frame_payload(data,result=None,title='空间刚架 · 三维分析'):
    if result is not None and not isinstance(result,dict):
        result=result.as_dict()
    raw=data.model_dump(mode='json') if isinstance(data,StructuralModel) else data
    model=StructuralModel.model_validate(raw)
    digest=model_hash(model)
    coords={node.id:np.array(node.coordinates) for node in model.nodes}
    span=max(float(np.ptp(np.array(list(coords.values())),axis=0).max()),1e-6)
    solved=result is not None
    if solved and digest!=result.get('model_hash'):
        raise ValueError('可视化模型与结果哈希不一致')
    display=textbook_presentation(result) if solved else None
    node_result={row['id']:row for row in display['nodes']} if solved else {}
    curves=sample_frame_results(model,result) if solved else []
    curve_lookup={row['id']:row for row in curves}
    end_result={row['id']:row for row in display['elements']} if solved else {}
    bases={}
    elements=[]
    nodes=[]
    for node in model.nodes:
        answer=node_result.get(node.id)
        displacement=answer['displacement'] if answer else None
        nodes.append({'id':node.id,'coordinates':list(node.coordinates),'restrained':list(node.restrained),
                      'displacement':displacement,'reaction':answer['reaction'] if answer else None,
                      'u_norm':float(np.linalg.norm(displacement[:3])) if answer else None})
    for element in model.elements:
        geometry,q=ch(coords[element.node_i],coords[element.node_j],element.local_y)
        bases[element.id]=(geometry[0],q)
        curve=curve_lookup.get(element.id)
        elements.append({'id':element.id,'node_i':element.node_i,'node_j':element.node_j,
                         'material_id':element.material_id,'section_id':element.section_id,
                         'length':float(geometry[0]),'basis':q.tolist(),
                         'start':coords[element.node_i].tolist(),'end':coords[element.node_j].tolist(),
                         'end_i':end_result[element.id]['end_i'] if solved else None,
                         'end_j':end_result[element.id]['end_j'] if solved else None,
                         'curve':curve})
    glyphs=[]
    loads=[]
    def add_load(label,kind,location,vector,description,unit):
        location=np.asarray(location,dtype=float)
        vector=np.asarray(vector,dtype=float)
        paths=moment_arc(location,vector,span*.045) if kind=='moment' else arrow(location,vector,span*.075)
        glyphs.append({'paths':paths,'kind':kind,'label':label+' · '+description,'position':location.tolist()})
        loads.append({'label':label,'kind':kind,'position':location.tolist(),
                      'global_components':vector.tolist(),'unit':unit,'description':description})
    unit='kN' if model.units=='kN_m' else 'N'
    for i,load in enumerate(model.loads,1):
        components=np.array(load.components)
        if np.linalg.norm(components[:3]):
            add_load(f'节点荷载{i}','force',coords[load.node_id],components[:3],f'节点{load.node_id}集中力',unit)
        if np.linalg.norm(components[3:]):
            add_load(f'节点力矩{i}','moment',coords[load.node_id],-components[3:],f'节点{load.node_id}力矩（右手）',unit+'·m')
    element_lookup={item.id:item for item in model.elements}
    for i,load in enumerate(model.member_loads,1):
        element=element_lookup[load.element_id]
        length,q=bases[element.id]
        vector=np.array(load.components)
        if load.coordinate_system=='local': vector=q@vector
        if load.kind=='uniform_force':
            a,b=load.interval
            paths=[]
            for x in np.linspace(a,b,9):
                tip=coords[element.node_i]+q[:,0]*x
                paths.extend(arrow(tip,vector,span*.055))
            glyphs.append({'kind':'distributed','paths':paths,'label':f'构件{element.id}均布力，区间[{a:g},{b:g}]m',
                           'position':(coords[element.node_i]+q[:,0]*(a+b)/2).tolist()})
            loads.append({'label':f'杆上荷载{i}','kind':'distributed','element_id':element.id,
                          'interval':[a,b],'global_components':vector.tolist(),'unit':unit+'/m',
                          'description':f'构件{element.id}分段均布荷载'})
        else:
            tip=coords[element.node_i]+q[:,0]*load.position
            moment=load.kind=='point_moment'
            add_load(f'杆上荷载{i}','moment' if moment else 'force',tip,-vector if moment else vector,
                     f'构件{element.id}，距i端{load.position:g}m',unit+'·m' if moment else unit)
    sections=[]
    section_rows=display['sections'] if solved else [
        {'id':query.id,'element_id':query.element_id,'location':query.location,
         'position':bases[query.element_id][0]/2 if query.location=='midspan' else query.position,
         'side':query.side,'forces':None} for query in model.section_queries]
    for row in section_rows:
        element=element_lookup[row['element_id']]
        _,q=bases[element.id]
        section=deepcopy(row)
        section['coordinates']=(coords[element.node_i]+q[:,0]*row['position']).tolist()
        sections.append(section)
    fields={}
    definitions=[('u_norm','位移模','m'),('ux','位移 ux','m'),('uy','位移 uy','m'),('uz','位移 uz','m'),
                 ('rx','转角 rx','rad'),('ry','转角 ry','rad'),('rz','转角 rz','rad'),
                 ('N','轴力 N',unit),('Qy','剪力 Qy',unit),('Qz','剪力 Qz',unit),
                 ('T','扭矩 T',unit+'·m'),('My','弯矩 My',unit+'·m'),('Mz','弯矩 Mz',unit+'·m')]
    for key,label,field_unit in definitions:
        values=[]
        for element in elements:
            if not solved: continue
            element['curve'].setdefault('fields',{})[key]=[]
            for sample in element['curve']['samples']:
                if key=='u_norm': value=float(np.linalg.norm(sample['displacement']))
                elif key in ('ux','uy','uz'): value=sample['displacement'][['ux','uy','uz'].index(key)]
                elif key in ('rx','ry','rz'): value=-sample['rotation_course'][['rx','ry','rz'].index(key)]
                else: value=sample['forces'][['N','Qy','Qz','T','My','Mz'].index(key)]
                element['curve']['fields'][key].append(value);values.append(value)
        maximum=max((abs(v) for v in values),default=0.)
        fields[key]={'label':label,'unit':field_unit,'min':0. if key=='u_norm' else -maximum,
                     'max':maximum if maximum else 1e-12,'data_abs_max':maximum,'all_zero':maximum==0,'available':solved,
                     'group':'displacement' if key in ('u_norm','ux','uy','uz','rx','ry','rz') else 'force'}
    max_node=max((node['u_norm'] for node in nodes if node['u_norm'] is not None),default=0.)
    max_curve=max((value for item in elements if solved for value in item['curve']['fields']['u_norm']),default=0.)
    amplification=min(1000.,max(1.,span*.1/max_curve)) if max_curve else 1.
    exports={}
    if solved:
        exports['nodes']=csv_text(['node_id','ux_m','uy_m','uz_m','rx_RH_rad','ry_RH_rad','rz_RH_rad'],
                                  [[n['id'],*n['displacement']] for n in nodes])
        exports['reactions']=csv_text(['node_id','Rx_'+unit,'Ry_'+unit,'Rz_'+unit,'Mx_RH_'+unit+'m','My_RH_'+unit+'m','Mz_RH_'+unit+'m'],
                                     [[n['id'],*n['reaction']] for n in nodes if any(n['restrained'])])
        exports['sections']=csv_text(['section_id','element_id','position_m','side','N_'+unit,'Qy_'+unit,'Qz_'+unit,'T_'+unit+'m','My_'+unit+'m','Mz_'+unit+'m'],
                                    [[s['id'],s['element_id'],s['position'],s['side'],*s['forces']] for s in sections])
    return {'title':title,'model_hash':digest,'solved':solved,'nodes':nodes,'elements':elements,
            'materials':model.model_dump(mode='json')['materials'],'section_properties':model.model_dump(mode='json')['sections'],
            'glyphs':glyphs,'loads':loads,'sections':sections,'fields':fields,'span':span,
            'force_unit':unit,'default_scale':amplification,'max_node_displacement_m':max_node,
            'max_sample_displacement_m':max_curve,'csv':exports,
            'notes':'节点/支座用全局坐标；杆端及截面内力用局部坐标。显示转角/力矩为附录右手，截面为局部截面正号。'}


def write_frame_html(model,result,path,title='空间刚架 · 三维分析'):
    payload=build_frame_payload(model,result,title)
    library=HERE/'assets/plotly-4.1.1.min.js'
    if not library.exists():
        raise ValueError('项目缺少 Plotly 离线资源，请恢复 src/visualization/assets')
    manifest=json.loads((HERE/'assets/manifest.json').read_text(encoding='utf-8'))
    code=library.read_text(encoding='utf-8')
    if hashlib.sha256(library.read_bytes()).hexdigest()!=manifest['sha256']:
        raise ValueError('Plotly 离线资源哈希不一致')
    serialized=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<',chr(92)+'u003c')
    template=(HERE/'viewer.html').read_text(encoding='utf-8')
    replacements={'TITLE':html.escape(title),'PLOTLY':code,'DATA':serialized}
    output=re.sub(r'__(TITLE|PLOTLY|DATA)__',lambda match:replacements[match.group(1)],template)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(output,encoding='utf-8')
    return {'path':str(path.resolve()),'model_hash':payload['model_hash'],'mode':'results' if result else 'structure'}
