"""材料/实心圆截面参数预处理；来源与推导见 docs/theory/material-section.md。"""
import hashlib
import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt
from src.model.structural import Material, Section, Number, Positive


class MaterialSectionInput(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    material_id: StrictInt
    section_id: StrictInt
    material_type: Literal['isotropic_linear_elastic']
    section_type: Literal['solid_circle']
    E: Positive
    E_unit: Literal['Pa', 'kPa', 'MPa', 'GPa', 'N/m^2', 'kN/m^2']
    nu: Annotated[Number, Field(gt=-1, lt=0.5)]
    R: Positive  # 半径，不是直径。
    R_unit: Literal['m', 'cm', 'mm']
    units: Literal['kN_m', 'N_m']
    source: Literal['image', 'user', 'textbook']
    evidence: str = Field(min_length=1)


class AxialRigidityInput(BaseModel):
    """桁架只给EA时的等效输入；A=1m²只作归一化，不代表真实截面。"""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    material_id: StrictInt
    section_id: StrictInt
    parameter_type: Literal['axial_rigidity']
    EA: Positive
    EA_unit: Literal['N', 'kN']
    units: Literal['kN_m', 'N_m']
    source: Literal['image', 'user', 'textbook']
    evidence: str = Field(min_length=1)


ParameterInput = MaterialSectionInput | AxialRigidityInput


def calculate_axial_rigidity(parameters):
    inputs = AxialRigidityInput.model_validate(parameters)
    EA = inputs.EA * (1000. if inputs.EA_unit == 'kN' else 1.)
    if inputs.units == 'kN_m': EA /= 1000.
    if not math.isfinite(EA) or EA <= 0:
        raise ValueError('EA单位换算产生无效结果')
    source = inputs.model_dump(mode='json')
    version = 'truss_axial_rigidity_v1'
    canonical = json.dumps({'formula_version':version,'input':source},sort_keys=True,
                           separators=(',',':'),allow_nan=False)
    return {'calculation_id':hashlib.sha256(canonical.encode()).hexdigest(),
        'formula_version':version,'input':source,'model_units':inputs.units,
        'material':{'id':inputs.material_id,'E':EA},'section':{'id':inputs.section_id,'A':1.},
        'units':{'E':'kN/m^2' if inputs.units=='kN_m' else 'N/m^2','A':'m^2',
                 'EA':'kN' if inputs.units=='kN_m' else 'N'},'axial_rigidity':EA,
        'assumptions':['仅适用于平面/空间桁架，保持题给EA',
                       '等效A=1m²、E=EA/A；不代表真实材料和截面，不能据此计算应力'],
        'theory':'docs/theory/planar-truss-frame-2014.md',
        'precision':'Python float全精度换算，保持EA乘积'}


PRESSURE_TO_PA = {'Pa': 1.0, 'kPa': 1e3, 'MPa': 1e6, 'GPa': 1e9,
                  'N/m^2': 1.0, 'kN/m^2': 1e3}
LENGTH_TO_M = {'m': 1.0, 'cm': 1e-2, 'mm': 1e-3}


def calculate_material_section(parameters):
    """原值/单位/来源 → Solver E/G/A/Iy/Iz/J；不依赖模型 API。"""
    raw = parameters.model_dump(mode='json') if isinstance(parameters, (MaterialSectionInput,AxialRigidityInput)) else parameters
    if isinstance(raw,dict) and raw.get('parameter_type') == 'axial_rigidity':
        return calculate_axial_rigidity(raw)
    inputs = MaterialSectionInput.model_validate(raw)
    try:
        E = inputs.E * PRESSURE_TO_PA[inputs.E_unit]
        if inputs.units == 'kN_m':
            E /= 1000.0
        R = inputs.R * LENGTH_TO_M[inputs.R_unit]
        # 参考资料例 IV-1，印刷 p186 / PDF p195：G=E/[2(1+ν)]。
        G = E / (2.0 * (1.0 + inputs.nu))
        # 同页：A=πR²，Jy=Jz=πD⁴/64，Jp=Jy+Jz；用 D=2R 写成半径形式。
        # Solver 名称 Iy/Iz/J 对应此圆截面的参考资料 Jy/Jz/Jp；一般截面不可套用。
        A = math.pi * R**2
        Iy = math.pi * R**4 / 4.0
        Iz = Iy
        J = math.pi * R**4 / 2.0
    except OverflowError:
        raise ValueError('参数换算或派生计算超出浮点数范围') from None
    if not all(math.isfinite(value) and value > 0 for value in (E, G, R, A, Iy, Iz, J)):
        raise ValueError('参数换算或派生计算产生非有限数或零值，请检查数值范围')
    material = Material(id=inputs.material_id, E=E, G=G).model_dump(mode='json')
    section = Section(id=inputs.section_id, A=A, Iy=Iy, Iz=Iz, J=J).model_dump(mode='json')
    source = inputs.model_dump(mode='json')
    version = 'solid_circle_isotropic_v1'
    canonical = json.dumps({'formula_version': version, 'input': source}, sort_keys=True,
                           separators=(',', ':'), allow_nan=False)
    calculation_id = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return {'calculation_id': calculation_id, 'formula_version': version,
            'input': source, 'model_units': inputs.units,
            'units': {'E': 'kN/m^2' if inputs.units == 'kN_m' else 'N/m^2',
                      'G': 'kN/m^2' if inputs.units == 'kN_m' else 'N/m^2',
                      'R': 'm', 'A': 'm^2', 'Iy': 'm^4', 'Iz': 'm^4', 'J': 'm^4'},
            'material': material, 'section': section, 'radius_m': R,
            'assumptions': ['各向同性线弹性材料', '实心圆截面', '截面惯性矩取形心轴',
                            'J 为 Saint-Venant 扭转常数，实心圆截面下等于极面积惯性矩'],
            'theory': 'docs/theory/material-section.md',
            'precision': 'Python float 全精度计算；参考资料表值的舍入仅用于参考比较，不回写求解参数'}


def format_parameters(result):
    """参考资料常用参数符号与单位；显示格式不改变原始计算值。"""
    values = {key: value for group in ('material', 'section') for key, value in result[group].items() if key != 'id'}
    if result['formula_version']=='truss_axial_rigidity_v1':
        return '\n'.join(f"{key} = {values[key]:.10g} {result['units'][key]}" for key in ('E','A'))+'\n仅为保持EA的归一化参数，不代表实际材性/截面。'
    return '\n'.join(f"{key} = {values[key]:.10g} {result['units'][key]}"
                     for key in ('E', 'G', 'A', 'Iy', 'Iz', 'J'))
