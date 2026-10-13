"""参数工具验证：解析值、参考资料舍入表值、单位与 Agent 注入边界。"""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from pydantic import ValidationError
from src.solver import MaterialSectionInput, calculate_material_section
from src.agent.runtime import FrameAgent
from .helpers import cantilever_data

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    return json.loads((ROOT/'examples/material_section_iv1.json').read_text(encoding='utf-8'))


def recognition_report():
    model = cantilever_data()
    model['nodes'][1]['restrained'] = [False]*6  # 辅助算例明确给定自由端，不使用 Agent 默认值。
    facts = []
    for key in ('units', 'convention'):
        facts.append({'path': '/'+key, 'value': model.get(key, 'course_clockwise'),
                      'state': 'user_given', 'source': 'user', 'evidence': '辅助算例明确给定'})
    for group in ('nodes', 'elements', 'materials', 'sections'):
        for i, row in enumerate(model[group]):
            for field, value in row.items():
                derived = group == 'materials' and field in ('E', 'G') or group == 'sections' and field in ('A', 'Iy', 'Iz', 'J')
                facts.append({'path': f'/{group}/{i}/{field}', 'value': None if derived else value,
                              'state': 'needs_calculation' if derived else 'user_given',
                              'source': 'user', 'evidence': '参数由工具计算；其他值由辅助算例给定'})
    return {'observations': ['辅助悬臂模型，用于参数注入验证；明确无荷载'], 'facts': facts,
            'loads': [], 'loads_complete': True, 'load_inventory_evidence': '辅助模型明确无荷载',
            'questions': [], 'requested_results': ['displacements']}


def agent():
    return FrameAgent(SimpleNamespace(config=SimpleNamespace(vision=False)))


class ParameterTests(unittest.TestCase):
    def test_exact_example_values(self):
        result = calculate_material_section(inputs())
        expected = {'E': 210000000.0, 'G': 80769230.76923077, 'A': 0.007853981633974483,
                    'Iy': 4.908738521234052e-6, 'Iz': 4.908738521234052e-6, 'J': 9.817477042468104e-6}
        for group in ('material', 'section'):
            for key, value in result[group].items():
                if key != 'id':
                    self.assertTrue(math.isclose(value, expected[key], rel_tol=1e-14, abs_tol=0), key)
        self.assertEqual(result['units']['G'], 'kN/m^2')

    def test_textbook_rounded_parameters(self):
        reference = json.loads((ROOT/'examples/textbook_iv1_original.json').read_text(encoding='utf-8'))
        result = calculate_material_section(inputs())
        # p186～187 参数表有舍入；此比较不把表值当全精度真值。
        # 归档参考的 J 由参考资料 Jp=Jy+Jz 及表中已舍入 Iy/Iz 得到，容差为两项之和。
        tolerances = {'E': 1e-8, 'G': 500.0, 'A': 5e-7, 'Iy': 5e-11, 'Iz': 5e-11, 'J': 1e-10}
        for group in ('material', 'section'):
            for key, value in result[group].items():
                if key != 'id':
                    self.assertLessEqual(abs(value-reference[key]), tolerances[key], key)

    def test_pressure_units(self):
        expected = calculate_material_section(inputs())['material']
        for unit, value in [('Pa', 210e9), ('N/m^2', 210e9), ('kPa', 210e6),
                            ('kN/m^2', 210e6), ('MPa', 210e3), ('GPa', 210.)]:
            with self.subTest(unit=unit):
                data = inputs()
                data.update(E=value, E_unit=unit)
                self.assertEqual(calculate_material_section(data)['material'], expected)

    def test_radius_units(self):
        expected = calculate_material_section(inputs())['section']
        for unit, value in [('m', 0.05), ('cm', 5.0), ('mm', 50.0)]:
            data = inputs()
            data.update(R=value, R_unit=unit)
            self.assertEqual(calculate_material_section(data)['section'], expected)

    def test_force_unit_scaling(self):
        data = inputs()
        kn = calculate_material_section(data)
        data['units'] = 'N_m'
        n = calculate_material_section(data)
        self.assertEqual(n['material']['E'], kn['material']['E']*1000)
        self.assertAlmostEqual(n['material']['G']/kn['material']['G'], 1000)
        self.assertEqual(n['section'], kn['section'])

    def test_invalid_input(self):
        changes = [{'E': 0}, {'E': -1}, {'R': 0}, {'R': -1}, {'nu': -1}, {'nu': .5},
                   {'nu': 1}, {'E': float('nan')}, {'R': float('inf')}, {'E_unit': 'psi'},
                   {'R_unit': 'inch'}, {'material_type': 'anisotropic'}, {'section_type': 'hollow_circle'},
                   {'E': True}, {'R': '0.05'}, {'material_id': True}, {'evidence': ' '}, {'diameter': .1}]
        for change in changes:
            with self.subTest(change=change):
                data = inputs()
                data.update(change)
                with self.assertRaises(ValidationError):
                    calculate_material_section(data)
        data = inputs()
        del data['nu']
        with self.assertRaises(ValidationError):
            calculate_material_section(data)

    def test_overflow_and_underflow(self):
        for change in [{'E': 1e308}, {'R': 1e100}, {'R': 1e-200}]:
            data = inputs()
            data.update(change)
            with self.assertRaises(ValueError):
                calculate_material_section(data)

    def test_mutated_input_is_revalidated(self):
        model = MaterialSectionInput.model_validate(inputs())
        model.nu = .7
        with self.assertRaises(ValidationError):
            calculate_material_section(model)

    def test_calculation_trace(self):
        data = inputs()
        original = deepcopy(data)
        first = calculate_material_section(data)
        second = calculate_material_section(data)
        self.assertEqual(first['calculation_id'], second['calculation_id'])
        self.assertEqual(first['input'], original)
        self.assertEqual(data, original)
        data['R'] = .06
        self.assertNotEqual(first['calculation_id'], calculate_material_section(data)['calculation_id'])

    def test_agent_injects_program_values(self):
        assistant = agent()
        report = recognition_report()
        self.assertEqual(assistant.record(report)['status'], 'needs_information')
        data = inputs()
        output = assistant.dispatch('calculate_material_section', {'parameters': data})
        self.assertEqual(output['status'], 'parameters_calculated')
        self.assertEqual(assistant.draft['status'], 'awaiting_confirmation')
        self.assertEqual(len(assistant.draft['computed_fields']), 6)
        self.assertEqual(assistant.model['materials'][0], output['result']['material'])
        self.assertEqual(assistant.model['sections'][0], output['result']['section'])
        # 原始识别记录仍为 null，派生值单独关联当前工具结果。
        values = {f['path']: f['value'] for f in assistant.draft['recognition']['facts']}
        self.assertIsNone(values['/materials/0/G'])
        self.assertIsNone(assistant.result)
        self.assertEqual(assistant.solve()['status'], 'needs_confirmation')

    def test_agent_blocks_conflicts_and_unit_mismatch(self):
        assistant = agent()
        assistant.calculate_parameters(inputs())
        report = recognition_report()
        for fact in report['facts']:
            if fact['path'] == '/materials/0/G':
                fact.update(value=1.0, state='user_given')
        output = assistant.record(report)
        self.assertIsNone(assistant.model)
        self.assertTrue(any('冲突' in issue for issue in output['blockers']))
        report = recognition_report()
        for fact in report['facts']:
            if fact['path'] == '/units':
                fact['value'] = 'N_m'
        output = assistant.record(report)
        self.assertTrue(any('单位不一致' in issue for issue in output['blockers']))

    def test_agent_keeps_uncertainty_and_unsupported_loads(self):
        assistant = agent()
        assistant.calculate_parameters(inputs())
        report = recognition_report()
        for fact in report['facts']:
            if fact['path'] == '/materials/0/G':
                fact['state'] = 'uncertain'
        report['loads'] = [{'label': 'q', 'kind': 'member_distributed', 'target_id': cantilever_data()['elements'][0]['id'],
                            'components': None, 'units': 'kN_m', 'position': None, 'interval': [0., 1.],
                            'raw_description': '仅用于验证不支持荷载仍保留', 'source': 'user', 'evidence': '用户明确给定'}]
        output = assistant.record(report)
        self.assertIsNone(output['candidate_model']['materials'][0]['G'])
        self.assertTrue(any('杆上荷载' in issue for issue in output['blockers']))
        self.assertIsNone(assistant.model)

    def test_invalid_recalculation_does_not_reuse_old_values(self):
        assistant = agent()
        assistant.record(recognition_report())
        assistant.calculate_parameters(inputs())
        data = inputs()
        data['nu'] = .8
        self.assertEqual(assistant.calculate_parameters(data)['status'], 'invalid_parameters')
        self.assertEqual(assistant.parameter_results, [])
        self.assertIsNone(assistant.model)
        self.assertEqual(assistant.record(recognition_report())['status'], 'needs_information')

    def test_new_user_message_invalidates_parameter_use(self):
        assistant = agent()
        assistant.calculate_parameters(inputs())
        assistant.record(recognition_report())
        assistant.client.complete = lambda *args: {'role': 'assistant', 'content': '等待新的参数工具调用'}
        assistant.chat('更改半径，请重新计算参数')
        self.assertEqual(len(assistant.parameter_results), 1)
        self.assertEqual(assistant.current_parameter_ids, set())
        self.assertIsNone(assistant.model)
        self.assertEqual(assistant.draft['status'], 'awaiting_update')
        # 保存历史计算不等于授权将它用于当前改变后的材料。
        self.assertEqual(assistant.record(recognition_report())['status'], 'needs_information')
