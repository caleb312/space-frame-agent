"""附录III.8例III-1原题与印刷答案的独立核对；不修改用户会话或Solver。"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from src.solver import solve_structure

ROOT=Path(__file__).resolve().parents[1]
SOURCE=dict(book='阎军、杨春秋《计算结构力学》2014',example='III-1',
            printed_pages=[168,169],pdf_pages=[177,178])
BOOK_U=np.array([.0000011,-.0000027,.0000204]) # 原印刷值保留，course_clockwise
BOOK_SECTIONS=np.array([[3.2485752,-5.3790646,2.3541472],
                       [-2.1657166,-.2118019,-.8472352],
                       [-8.1672621,-5.4142919,2.3541472]])


def original_model():
    # ②杆按教材3→4；真实均布力仍全局-y。type6的VOL=-5是书中荷载类型输入，
    # 不等于节点课程Mz=-5。p176固定端反作用把端点顺时针力偶传为节点Mz=+5。
    return dict(structure_type='plane_frame',units='kN_m',convention='course_clockwise',
        nodes=[dict(id=i,coordinates=p,restrained=fix) for i,p,fix in [
            (1,[2,0],[True]*3),(2,[0,2],[True]*3),(3,[5,2],[True]*3),(4,[2,2],[False]*3)]],
        elements=[dict(id=i,node_i=a,node_j=b,material_id=1,section_id=1) for i,a,b in [(1,2,4),(2,3,4),(3,1,4)]],
        materials=[dict(id=1,E=200000000)],sections=[dict(id=1,A=.1,I=.000005)],
        loads=[dict(node_id=4,components=[0,0,5])],member_loads=[
            dict(element_id=1,kind='point_force',coordinate_system='global',components=[0,-10,0],position=1),
            dict(element_id=2,kind='uniform_force',coordinate_system='global',components=[0,-2,0],interval=[0,3]),
            dict(element_id=3,kind='point_force',coordinate_system='global',components=[10,0,0],position=1)],
        section_queries=[dict(id=k,element_id=k,location='midspan',side='right') for k in (1,2,3)])


def free_equations(A=.1,I=.000005):
    """独立手列节点4三个未知量；采用课程顺时针坐标。"""
    EA=200000000*A;EI=200000000*I
    K=np.array([[EA/2+EA/3+12*EI/2**3,0,-6*EI/2**2],
        [0,EA/2+12*EI/2**3+12*EI/3**3,6*EI/2**2-6*EI/3**2],
        [-6*EI/2**2,6*EI/2**2-6*EI/3**2,4*EI/2+4*EI/3+4*EI/2]])
    return K,np.array([5.,-8.,1.5])


def ldlt_float32(A,b):
    """按教材p176-177的LDLT/回代顺序在float32计算，检查浮点量级。"""
    A=np.asarray(A,np.float32);b=np.asarray(b,np.float32);n=len(b)
    L=np.eye(n,dtype=np.float32);D=np.zeros(n,np.float32);y=np.zeros(n,np.float32);x=np.zeros(n,np.float32)
    D[0]=A[0,0]
    for i in range(1,n):
        for j in range(i):
            s=np.float32(0)
            for k in range(j):s=np.float32(s+L[i,k]*D[k]*L[j,k])
            L[i,j]=(A[i,j]-s)/D[j]
        s=np.float32(0)
        for j in range(i):s=np.float32(s+L[i,j]*L[i,j]*D[j])
        D[i]=A[i,i]-s
    y[0]=b[0]
    for i in range(1,n):
        s=np.float32(0)
        for k in range(i):s=np.float32(s+L[i,k]*y[k])
        y[i]=b[i]-s
    x[-1]=y[-1]/D[-1]
    for i in range(n-2,-1,-1):
        s=np.float32(0)
        for k in range(i+1,n):s=np.float32(s+L[k,i]*x[k])
        x[i]=y[i]/D[i]-s
    return x.astype(float)


def compare(model):
    result=solve_structure(model).as_dict()
    u=np.array(next(n['native_displacement'] for n in result['nodes'] if n['id']==4))
    sections=np.array([[q['forces'][i] for i in (0,1,5)] for q in result['sections']])
    return result,u,sections


def audit():
    model=original_model();result,u,f=compare(model);K,P=free_equations()
    independent=np.linalg.solve(K,P)
    np.testing.assert_allclose(u,independent,atol=1e-14,rtol=1e-10)
    alternate=deepcopy(model);alternate['sections'][0].update(A=.03,I=.00007162)
    alternate_result,alt_u,alt_f=compare(alternate)
    float32_tolerance=np.abs(np.spacing(np.abs(BOOK_SECTIONS).astype(np.float32))).astype(float)+5e-8
    assert np.all(np.abs(alt_f-BOOK_SECTIONS)<=float32_tolerance)
    assert [f'{v:.7f}' for v in alt_u]==[f'{v:.7f}' for v in BOOK_U]
    return dict(source=SOURCE,original_book_displacement_course=BOOK_U.tolist(),
        original_book_sections_N_Q_M=BOOK_SECTIONS.tolist(),original_model=model,
        original_model_hash=result['model_hash'],computed_displacement_course=u.tolist(),
        computed_sections_N_Q_M=f.tolist(),independent_free_K=K.tolist(),independent_free_P=P.tolist(),
        independent_displacement_course=independent.tolist(),independent_residual=(K@u-P).tolist(),
        printed_displacement_residual=(K@BOOK_U-P).tolist(),
        printed_displacement_rounding_residual_bound=(np.abs(K)@np.full(3,5e-8)).tolist(),
        float32_displacement_course=ldlt_float32(K,P).tolist(),
        float32_difference=(ldlt_float32(K,P)-u).tolist(),
        stated_vs_book_max_force_error_kN=float(np.max(abs(f[:,:2]-BOOK_SECTIONS[:,:2]))),
        stated_vs_book_max_moment_error_kNm=float(np.max(abs(f[:,2]-BOOK_SECTIONS[:,2]))),
        alternate_section_A_m2=.03,alternate_section_I_m4=.00007162,
        alternate_section_source='同页下一例III-2题面及输入表，印刷p169/PDFp178',
        alternate_displacement_course=alt_u.tolist(),alternate_sections_N_Q_M=alt_f.tolist(),
        alternate_max_force_error_kN=float(np.max(abs(alt_f[:,:2]-BOOK_SECTIONS[:,:2]))),
        alternate_max_moment_error_kNm=float(np.max(abs(alt_f[:,2]-BOOK_SECTIONS[:,2]))),
        displacement_matches_printed_7_decimal_places=True,
        all_sections_match_float32_and_print_rounding_tolerance=True,
        conclusion='原题截面参数与印刷结果不匹配；只用下一例截面参数即复现印刷结果。未改变实际工作台条件。')


def main():
    report=audit()
    (ROOT/'results/textbook_iii1_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 附录III.8例III-1核对','',
        '教材：阎军、杨春秋2014《计算结构力学》，印刷p168–169 / PDFp177–178。',
        '原题和答案均按教材原页转录，不覆盖原印刷值。当前工作台条件与结果未被修改。','',
        '## 结论','',report['conclusion'],
        '不是当前Solver浮点舍入造成的差异。原题独立节点4方程与Solver一致；float32复算误差不足以解释原答案差异。','',
        '| 参数 | 例III-1题面/输入 | 可复现答案的参数（与下一例III-2相同） |',
        '| --- | --- | --- |','| E / kN·m⁻² | 200000000 | 200000000 |',
        '| A / m² | 0.1 | 0.03 |','| I / m⁴ | 0.000005 | 0.00007162 |','',
        '## 节点4（课程顺时针转角为正）','',
        '| 分量 | 原印刷 | 原题正确计算 | 仅换下一例截面参数 |','| --- | --- | --- |']
    for i,key in enumerate(['ux / m','uy / m','rz / rad']):
        lines.append(f'| {key} | {BOOK_U[i]:.7f} | {report["computed_displacement_course"][i]:.12g} | {report["alternate_displacement_course"][i]:.12g} |')
    lines+=['','工作台显示转角为右手正号，因此原题rz显示值为上述课程rz的负值。','',
        '## 跨中截面（教材局部轴、教材截面正号）','',
        '②杆按教材i=3,j=4统一方向；工作台原i=4,j=3的跨中弯矩需反号比较。N/Q并非简单全部反号。',
        '①、③的集中力正落截面，采用right极限，与教材程序严格小于判断所对应的切面一致。','',
        '| 截面 | 项目 | 原印刷 | 原题计算 | 仅换截面参数 |','| --- | --- | --- | --- | --- |']
    for k in range(3):
        for j,key in enumerate(['N / kN','Q / kN','M / kN·m']):
            lines.append(f'| {k+1} | {key} | {BOOK_SECTIONS[k,j]:.7f} | {report["computed_sections_N_Q_M"][k][j]:.10g} | {report["alternate_sections_N_Q_M"][k][j]:.10g} |')
    lines+=['','仅换截面参数后，节点三项位移按教材7位小数完全一致，9项内力均在float32一个ULP加印刷半末位容差内。',
        f'最大力差{report["alternate_max_force_error_kN"]:.6g} kN；最大弯矩差{report["alternate_max_moment_error_kNm"]:.6g} kN·m。',
        '这说明答案数值对应另一组截面参数；无法仅凭扫描本确定错在题面参数还是答案编辑过程，不声称出版社正式勘误。',
        '此前Agent把顺时针误填Mz=-5是另外一个输入错误，已修正；本核对全部使用正确的Mz=+5。',
        '完整矩阵、印刷位移残差、float32误差及原始转录在同名JSON。核心Solver保持不变。']
    (ROOT/'results/textbook_iii1_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['computed_displacement_course','alternate_displacement_course',
        'stated_vs_book_max_force_error_kN','alternate_max_force_error_kN','alternate_max_moment_error_kNm',
        'printed_displacement_residual','printed_displacement_rounding_residual_bound','float32_difference']},indent=2))


if __name__=='__main__':main()
