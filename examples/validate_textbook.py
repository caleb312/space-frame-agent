"""复现附录IV两道空间算例；跨内荷载转换和符号转换均显式记录。"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.solver import solve_space_frame
from .textbook_reference import (equivalent_model, displacements_in_appendix_convention,
                                  original_end_forces_right_hand, section_forces_in_appendix_convention)


def validate_case(name):
    root=Path(__file__).resolve().parents[1]
    raw=json.loads((root/f'examples/textbook_{name}_original.json').read_text(encoding='utf-8'))
    model,equivalent,_=equivalent_model(raw)
    saved=json.loads((root/f'examples/textbook_{name}_equivalent_nodes.json').read_text(encoding='utf-8'))
    if saved != model:
        raise ValueError(f'{name}: 保存的等效节点模型与原题转换不一致')
    result=solve_space_frame(model)
    all_displacements=displacements_in_appendix_convention(result)
    index={node_id:i for i,node_id in enumerate(result.arrays.node_ids)}
    actual=np.array([all_displacements[index[node_id]] for node_id in raw['reference_node_ids']])
    original=np.array(raw['reference_displacements'])
    comparison=np.array(raw.get('reference_displacements_corrected',raw['reference_displacements']))
    np.testing.assert_allclose(actual[:,:3],comparison[:,:3],atol=raw['displacement_atol'],rtol=0)
    np.testing.assert_allclose(actual[:,3:],comparison[:,3:],atol=raw['rotation_atol'],rtol=0)
    sections=section_forces_in_appendix_convention(raw,result,equivalent)
    section_reference=np.array(raw['reference_section_forces'])
    np.testing.assert_allclose(sections[:,:3],section_reference[:,:3],atol=1e-3,rtol=0)
    np.testing.assert_allclose(sections[:,3:],section_reference[:,3:],atol=1e-3,rtol=0)
    return {
        'source':raw['source'],'comparison_passed':True,
        'scope':'Original textbook loads explicitly converted to equivalent nodes for core Solver; original section forces recovered by benchmark adapter',
        'model_hash':result.model_hash,'reference_node_ids':raw['reference_node_ids'],
        'actual_displacements_appendix_right_hand':actual.tolist(),
        'printed_displacements':original.tolist(),'comparison_displacements':comparison.tolist(),
        'printed_displacements_match':bool(np.all(np.abs(actual[:,:3]-original[:,:3])<=raw['displacement_atol'])
                                          and np.all(np.abs(actual[:,3:]-original[:,3:])<=raw['rotation_atol'])),
        'erratum':raw.get('project_errata'),
        'max_displacement_abs_error':float(np.max(np.abs(actual[:,:3]-comparison[:,:3]))),
        'max_rotation_abs_error':float(np.max(np.abs(actual[:,3:]-comparison[:,3:]))),
        'sections':raw['sections'],'actual_section_forces_appendix':sections.tolist(),
        'printed_section_forces':section_reference.tolist(),
        'max_section_force_abs_error':float(np.max(np.abs(sections[:,:3]-section_reference[:,:3]))),
        'max_section_moment_abs_error':float(np.max(np.abs(sections[:,3:]-section_reference[:,3:]))),
        'original_end_forces_right_hand':original_end_forces_right_hand(result,equivalent).tolist(),
        'support_reactions_course':result.reactions.reshape(-1,6).tolist(),
        'tolerances':{'displacement_m':raw['displacement_atol'],'rotation_rad':raw['rotation_atol'],
                      'section_force_kN':1e-3,'section_moment_kNm':1e-3},
    }


def main():
    parser=argparse.ArgumentParser(description='项目计算规则附录IV原题复现与明确勘误核对')
    parser.add_argument('--output',type=Path,default=Path('results/textbook_verification.json'))
    args=parser.parse_args()
    cases={name:validate_case(name) for name in ('iv1','iv2')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(cases,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,case in cases.items():
        print(f"{name}: 对照通过；位移误差 {case['max_displacement_abs_error']:.3e} m；"
              f"转角误差 {case['max_rotation_abs_error']:.3e} rad；"
              f"截面力误差 {case['max_section_force_abs_error']:.3e} kN")
        if case['erratum']:
            print('  IV-2 原印刷 x 位移列未通过原样对照；使用已保留原值的显式项目勘误 E04。')


if __name__=='__main__':
    main()
