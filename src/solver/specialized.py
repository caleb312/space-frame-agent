"""项目计算规则原生平桁/空桁/平刚：ch→stif→dor→ldlt→slveq→doh。"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from src.model.specialized import SpecializedModel
from .equations import SolverError, StabilityError, ldlt, slveq
from .member_loads import LocalLoad, consistent_vector, section_forces


def ch_native(xi,xj):
    delta=np.asarray(xj,float)-np.asarray(xi,float)
    L=float(np.linalg.norm(delta))
    return L,delta/L


def stif_truss(EA,L,n):
    """p32、p70：T(EA/L)Tᵀ，2/3维轴力杆，同一方向余弦定义。"""
    block=EA/L*np.outer(n,n)
    return np.block([[block,-block],[-block,block]])


def stif_plane_frame(EA,EI,L):
    """p78四分块，局部[ux,uy,顺时针rz]。"""
    a,b,c,d,e=EA/L,12*EI/L**3,6*EI/L**2,4*EI/L,2*EI/L
    aa=np.array([[a,0,0],[0,b,-c],[0,-c,d]])
    ab=np.array([[-a,0,0],[0,-b,-c],[0,c,e]])
    bb=np.array([[a,0,0],[0,b,c],[0,c,d]])
    return np.block([[aa,ab],[ab.T,bb]])


def local_basis(n):
    ex=np.r_[n,0.0] if len(n)==2 else n.copy()
    if len(n)==2:
        ey=np.array([-n[1],n[0],0.0])
    else:
        hint=np.array([0.,0.,1.]) if abs(ex[2])<.9 else np.array([0.,1.,0.])
        ey=hint-ex*np.dot(ex,hint);ey/=np.linalg.norm(ey)
    return np.column_stack((ex,ey,np.cross(ex,ey)))


@dataclass
class NativeResult:
    output: dict
    K: np.ndarray
    P: np.ndarray
    u: np.ndarray
    reactions: np.ndarray
    free: np.ndarray
    factors: object

    def as_dict(self,diagnostics=False):
        value=deepcopy(self.output)
        if diagnostics:
            value['native_matrices']={'K_full':self.K.tolist(),'P_full':self.P.tolist(),
                'U_full':self.u.tolist(),'free_indices':self.free.tolist(),
                'L':self.factors.L.tolist(),'D':self.factors.D.tolist()}
        return value


def solve_native(model,expected_type):
    raw=model.model_dump(mode='json') if hasattr(model,'model_dump') else model
    model=SpecializedModel.model_validate(raw)
    if model.structure_type!=expected_type:
        raise SolverError('求解入口与结构类型不一致')
    frame=expected_type=='plane_frame';dim=model.dimension;nd=model.dof_count
    ids={node.id:index for index,node in enumerate(model.nodes)}
    nodes={node.id:node for node in model.nodes};mats={m.id:m for m in model.materials};secs={s.id:s for s in model.sections}
    K=np.zeros((nd*len(nodes),nd*len(nodes)));P=np.zeros(nd*len(nodes))
    restrained=np.array([n.restrained for n in model.nodes],bool).ravel()
    elements={};native_end={};local_end={}
    for load in model.loads:
        index=ids[load.node_id]*nd;P[index:index+nd]+=load.components
    # dor：显式i/j对号与单元四分块组装，无旋转伪自由度。
    for e in model.elements:
        L,n=ch_native(nodes[e.node_i].coordinates,nodes[e.node_j].coordinates)
        EA=mats[e.material_id].E*secs[e.section_id].A
        dofs=np.r_[np.arange(nd)+nd*ids[e.node_i],np.arange(nd)+nd*ids[e.node_j]]
        Q=local_basis(n)
        if frame:
            local=stif_plane_frame(EA,mats[e.material_id].E*secs[e.section_id].I,L)
            T=np.zeros((6,6))
            # 2D force transform has translations rotated in XY and scalar rz unchanged.
            R=np.array([[n[0],-n[1],0],[n[1],n[0],0],[0,0,1]])
            T[:3,:3]=R;T[3:,3:]=R
            global_k=T@local@T.T
        else:
            global_k=stif_truss(EA,L,n);local=None;T=None
        K[np.ix_(dofs,dofs)]+=global_k
        elements[e.id]={'L':L,'n':n,'Q':Q,'EA':EA,'local':local,'T':T,'dofs':dofs,
                         'equivalent':np.zeros(6),'loads':[]}
    indices=[0,1,5,6,7,11]
    for load in model.member_loads:
        item=elements[load.element_id];components=np.asarray(load.components,float)
        if load.kind=='point_moment':
            vec=np.array([0.,0.,components[2]])
        else:
            vec=np.array([components[0],components[1],0.])
            if load.coordinate_system=='global':vec=item['Q'].T@vec
        local_load=LocalLoad(load.element_id,load.kind,vec,load.position,load.interval)
        item['loads'].append(local_load)
        item['equivalent']+=consistent_vector(item['L'],local_load)[indices]
    if frame:
        for item in elements.values():P[item['dofs']]+=item['T']@item['equivalent']
    free=np.flatnonzero(~restrained);r=K[np.ix_(free,free)]
    try:factors=ldlt(r)
    except StabilityError as error:
        at=int(free[error.equation]);dofs=['ux','uy','rz'] if frame else ['ux','uy','uz'][:dim]
        raise SolverError(f'{error}；节点{model.nodes[at//nd].id}/{dofs[at%nd]}') from error
    u=np.zeros(len(P));u[free]=slveq(factors,P[free])
    reactions=K@u-P
    norm=float(np.linalg.norm(r,np.inf)*np.linalg.norm(u[free],np.inf)+np.linalg.norm(P[free],np.inf)) if len(free) else 0.
    residual=float(np.max(np.abs(reactions[free]))) if len(free) else 0.
    backward=residual/norm if norm else residual
    if not np.all(np.isfinite(u)) or backward>1e-9:
        raise SolverError('原生求解结果非有限或方程残差超限')
    for e in model.elements:
        item=elements[e.id];ue=u[item['dofs']]
        if frame:
            h=item['local']@(item['T'].T@ue)-item['equivalent']
            hi=np.array([h[0],h[1],0,0,0,h[2]]);hj=np.array([h[3],h[4],0,0,0,h[5]])
            N=-h[0]
        else:
            N=item['EA']/item['L']*np.dot(item['n'],ue[dim:]-ue[:dim])
            h=np.array([-N,N]);hi=np.array([-N,0,0,0,0,0]);hj=-hi
        native_end[e.id]=h;local_end[e.id]=(hi,hj,N)
    queries=[]
    for q in model.section_queries:
        item=elements[q.element_id];pos=item['L']/2 if q.location=='midspan' else q.position
        values=section_forces(local_end[q.element_id][0],item['loads'],pos,q.side)
        queries.append({'id':q.id,'element_id':q.element_id,'position':pos,'side':q.side,
                        'location':q.location,'forces':values.tolist()})
    # 独立按原始节点/杆上荷载计算整体平衡，不以等效节点力冒充原题荷载。
    force=np.zeros(3);moment=np.zeros(3)
    def point(x):return np.r_[x,0.] if len(x)==2 else np.asarray(x,float)
    for load in model.loads:
        values=np.asarray(load.components);f=np.r_[values[:2],0.] if dim==2 else values[:3]
        force+=f;moment+=np.cross(point(nodes[load.node_id].coordinates),f)
        if frame:moment[2]-=values[2]
    for load in model.member_loads:
        item=elements[load.element_id];e=next(e for e in model.elements if e.id==load.element_id)
        if load.kind=='point_moment':moment[2]-=load.components[2];continue
        vec=np.array([*load.components[:2],0.]);vec=item['Q']@vec if load.coordinate_system=='local' else vec
        s=load.position
        if load.kind=='uniform_force':
            a,b=load.interval;vec*=b-a;s=(a+b)/2
        force+=vec;moment+=np.cross(point(nodes[e.node_i].coordinates)+item['Q'][:,0]*s,vec)
    active=[0,1,5] if frame else list(range(dim));out_nodes=[]
    for index,node in enumerate(model.nodes):
        un=u[index*nd:(index+1)*nd];rn=reactions[index*nd:(index+1)*nd]
        padded_u=np.zeros(6);padded_r=np.zeros(6);padded_fix=np.zeros(6,bool)
        padded_u[active]=un;padded_r[active]=rn;padded_fix[active]=node.restrained
        force+=padded_r[:3];moment+=np.cross(point(node.coordinates),padded_r[:3])-padded_r[3:]
        out_nodes.append({'id':node.id,'displacement':padded_u.tolist(),'reaction':padded_r.tolist(),
            'restrained':padded_fix.tolist(),'native_displacement':un.tolist(),'native_reaction':rn.tolist(),
            'native_restrained':node.restrained})
    canonical=json.dumps(model.model_dump(mode='json'),sort_keys=True,separators=(',',':'),allow_nan=False)
    unit='kN' if model.units=='kN_m' else 'N'
    output={'model_hash':hashlib.sha256(canonical.encode()).hexdigest(),'structure_type':expected_type,
        'convention':'course_clockwise','native_dof_order':['ux','uy','rz'] if frame else ['ux','uy','uz'][:dim],
        'display_padding_note':'展示六分量中的非原生自由度为占位零，不是额外支座约束',
        'dof_order':['ux','uy','uz','rx','ry','rz'],'end_force_order':['Fx','Fy','Fz','Mx','My','Mz'],
        'section_force_order':['N','Qy','Qz','T','My','Mz'],'section_force_convention':'textbook_section',
        'units':{'displacement':'m','rotation':'rad','force':unit,'moment':unit+'*m','distributed_force':unit+'/m'},
        'nodes':out_nodes,'elements':[{'id':e.id,'node_i':e.node_i,'node_j':e.node_j,
            'end_i':local_end[e.id][0].tolist(),'end_j':local_end[e.id][1].tolist(),
            'axial_force':local_end[e.id][2],'local_basis_global':elements[e.id]['Q'].tolist(),
            'native_end_forces':native_end[e.id].tolist()} for e in model.elements],
        'sections':queries,'member_loads':[load.model_dump(mode='json') for load in model.member_loads],
        'checks':{'backward_error':backward,'free_residual':reactions[free].tolist(),
            'force_balance':force.tolist(),'right_hand_moment_balance':moment.tolist()}}
    return NativeResult(output,K,P,u,reactions,free,factors)


def solve_plane_truss(model):return solve_native(model,'plane_truss')
def solve_space_truss(model):return solve_native(model,'space_truss')
def solve_plane_frame(model):return solve_native(model,'plane_frame')
