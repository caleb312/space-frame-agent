"""用已求解的杆端力和原荷载恢复杆内位移/转角；不重新求解 K U=P。"""
import hashlib
import json

import numpy as np
from src.model import StructuralModel
from .geometry import ch
from .member_loads import LocalLoad, section_forces


def sample_frame_results(model, result, count=65):
    """返回各杆的真实变形采样与左右截面内力，原数据不修改。"""
    if not isinstance(result,dict):
        result=result.as_dict()
    data=model.model_dump(mode='json') if isinstance(model,StructuralModel) else model
    model=StructuralModel.model_validate(data)
    if result.get('convention')!='course_clockwise':
        raise ValueError('后处理需要原始课件约定结果')
    canonical=json.dumps(model.model_dump(mode='json'),sort_keys=True,separators=(',',':'),allow_nan=False)
    if hashlib.sha256(canonical.encode('utf-8')).hexdigest()!=result.get('model_hash'):
        raise ValueError('可视化模型与计算结果哈希不一致')
    if type(count) is not int or not 3<=count<=301:
        raise ValueError('每杆采样数应为 3～301')
    nodes={node.id:node for node in model.nodes}
    answers={item['id']:item for item in result['nodes']}
    ends={item['id']:item for item in result['elements']}
    materials={item.id:item for item in model.materials}
    sections={item.id:item for item in model.sections}
    points,weights=np.polynomial.legendre.leggauss(3)
    output=[]
    for element in model.elements:
        xi=np.array(nodes[element.node_i].coordinates)
        xj=np.array(nodes[element.node_j].coordinates)
        geometry,q=ch(xi,xj,element.local_y)
        length=geometry[0]
        mat,sec=materials[element.material_id],sections[element.section_id]
        u0=np.array(answers[element.node_i]['displacement'])
        u1=np.array(answers[element.node_j]['displacement'])
        if u0.shape!=(6,) or u1.shape!=(6,) or not np.all(np.isfinite(np.r_[u0,u1])):
            raise ValueError('后处理的节点位移无效')
        local0=np.r_[q.T@u0[:3],q.T@u0[3:]]
        end_i=np.array(ends[element.id]['end_i'])
        loads=[]
        breaks={0.,float(length)}
        discontinuities=set()
        for item in model.member_loads:
            if item.element_id!=element.id:
                continue
            components=np.array(item.components)
            if item.coordinate_system=='global':
                components=q.T@components
            loads.append(LocalLoad(element.id,item.kind,components,item.position,item.interval))
            if item.kind=='uniform_force':
                breaks.update(item.interval)
            else:
                breaks.add(item.position)
                discontinuities.add(item.position)
        def displacement(x):
            integral=np.zeros(6)  # ∫N/EA，∫Mz/EIz，∫My/EIy，∫T/GJ，∫(x-s)Mz/EIz，∫(x-s)My/EIy。
            boundaries=sorted({0.,float(x),*(v for v in breaks if 0<v<x)})
            for a,b in zip(boundaries[:-1],boundaries[1:]):
                for p,w in zip(points,weights):
                    s=(a+b)/2+(b-a)*p/2
                    force=section_forces(end_i,loads,s,'right')
                    terms=np.array([force[0]/(mat.E*sec.A), force[5]/(mat.E*sec.Iz),
                                    force[4]/(mat.E*sec.Iy), force[3]/(mat.G*sec.J),
                                    (x-s)*force[5]/(mat.E*sec.Iz), (x-s)*force[4]/(mat.E*sec.Iy)])
                    integral+=w*(b-a)/2*terms
            local_u=np.array([local0[0]+integral[0],local0[1]-local0[5]*x+integral[4],
                              local0[2]+local0[4]*x+integral[5]])
            local_r=np.array([local0[3]-integral[3],local0[4]+integral[2],local0[5]-integral[1]])
            return q@local_u,q@local_r
        du,dr=displacement(length)
        u_error=float(np.max(np.abs(du-u1[:3])))
        r_error=float(np.max(np.abs(dr-u1[3:])))
        if u_error>1e-8*max(1.,float(np.max(np.abs(u1[:3])))) or r_error>1e-8*max(1.,float(np.max(np.abs(u1[3:])))):
            raise ValueError(f'构件 {element.id}：杆内变形与节点结果不一致，不能绘图')
        samples=set(np.linspace(0,length,count).tolist())|breaks
        samples.update(item['position'] for item in result.get('sections',[]) if item['element_id']==element.id)
        rows=[]
        for x in sorted(samples):
            d,r=displacement(x)
            for side in (('left','right') if x in discontinuities else ('right',)):
                rows.append({'position':float(x),'side':side,'coordinates':(xi+q[:,0]*x).tolist(),
                             'displacement':d.tolist(),'rotation_course':r.tolist(),
                             'forces':section_forces(end_i,loads,x,side).tolist()})
        output.append({'id':element.id,'node_i':element.node_i,'node_j':element.node_j,
                       'length':float(length),'basis':q.tolist(),'samples':rows,
                       'endpoint_error_m':u_error,'endpoint_error_rad':r_error})
    return output
