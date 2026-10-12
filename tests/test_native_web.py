"""原生模型的工作台确认、保存/恢复与纠正失效；不请求真实API。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.web import server
from src.agent.providers import APIError
from src.agent.runtime import fingerprint
from src.solver import solve_structure
from examples.prepare_midterm_examples import make,recognition


def cases():
    return [
        make('plane_truss',[[0,0],[2,0],[1,1]],
             [[True,True],[False,True],[False,False]],[(1,2),(1,3),(2,3)],[(3,[0,-10])]),
        make('plane_frame',[[0,0],[2,0]],[[True]*3,[False]*3],[(1,2)],[(2,[0,-10,0])]),
        make('space_truss',[[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
             [[True]*3]*3+[[False]*3],[(1,4),(2,4),(3,4)],[(4,[3,-4,5])])]


class ImmediateThread:
    """保留run_action/worker流程，测试中同步执行便于检查稳定状态。"""
    def __init__(self,target,args,daemon):self.target,self.args=target,args
    def start(self):self.target(*self.args)


class NativeWebTests(unittest.TestCase):
    def setUp(self):
        work=(Path(__file__).resolve().parents[1]/'work').resolve()
        work.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='frame_native_web_',dir=work)
        self.root=Path(self.temp.name).resolve()
        if not self.root.is_relative_to(work):raise ValueError('测试临时目录越界')
        self.addCleanup(self.temp.cleanup)
        (self.root/'results').mkdir()
        for target,value in [('ROOT',self.root),('STORE',self.root/'results'/'web_sessions')]:
            p=patch.object(server,target,value);p.start();self.addCleanup(p.stop)
        p=patch.object(server,'defaults',return_value={'provider':'qwen','model':'test',
            'base_url':'https://example.invalid','vision':True,'thinking':'disabled',
            'timeout':1,'max_tokens':1024,'environment_key_available':False})
        p.start();self.addCleanup(p.stop)

    def session(self,index=0):return server.Session(f'{index:032x}','qwen')

    def test_confirm_solve_and_recover_three_native_types(self):
        for index,raw in enumerate(cases(),1):
            s=self.session(index)
            draft=s.agent.record(recognition(raw))
            self.assertEqual(draft['status'],'awaiting_confirmation')
            self.assertIsNone(s.agent.result)
            with self.assertRaisesRegex(ValueError,'模型已变化'):
                s.run_action('solve',{'model_hash':'old'})
            with patch.object(server.threading,'Thread',ImmediateThread):
                s.run_action('solve',{'model_hash':fingerprint(s.agent.model),'explain':True})
            state=s.public()
            self.assertEqual(state['status'],'solved',state['error'])
            self.assertFalse(state['busy'])
            self.assertIsNotNone(state['visual'])
            self.assertEqual(s.agent.events[-1]['tool'],'solve_'+raw['structure_type'])
            self.assertTrue(state['answer'])
            self.assertEqual(len(state['result']['native_dof_order']),2 if index==1 else 3)
            restored=self.session(index)
            self.assertEqual(restored.public()['status'],'solved',restored.error)
            self.assertEqual(restored.agent.result,s.agent.result)
            self.assertIsNone(restored.agent.confirmed_hash)
            self.assertFalse(restored.public()['settings']['connected'])

    def test_open_saved_native_models_without_recalculation(self):
        for index,raw in enumerate(cases(),1):
            s=self.session(index);draft=s.agent.record(recognition(raw))
            result=solve_structure(s.agent.model).as_dict()
            name=raw['structure_type']+'.json'
            (self.root/'results'/name).write_text(json.dumps({'model':s.agent.model,'result':result}),encoding='utf-8')
            (self.root/'results'/name.replace('.json','.recognition.json')).write_text(
                json.dumps({'draft':draft}),encoding='utf-8')
            self.assertIn(name,{item['name'] for item in server.saved_results()})
            with patch('src.agent.runtime.solve_structure',side_effect=AssertionError('不应重新求解')):
                s.open_saved(name)
            self.assertEqual(s.public()['status'],'solved')
            self.assertEqual(s.agent.result,result)
            self.assertEqual(s.agent.draft['recognition']['structure_type'],raw['structure_type'])
            self.assertIsNotNone(s.public()['visual'])
            text=(s.path/'summary.md').read_text(encoding='utf-8')
            self.assertIn('节点位移',text)
            if raw['structure_type']!='plane_frame':self.assertNotIn('Mz（右手）',text)

    def test_failed_correction_revokes_old_result_and_confirmation(self):
        s=self.session();s.agent.record(recognition(cases()[0]))
        with patch.object(server.threading,'Thread',ImmediateThread):
            s.run_action('solve',{'model_hash':fingerprint(s.agent.model)})
        class FailingClient:
            config=SimpleNamespace(vision=True)
            def complete(self,*args):raise APIError('本测试模拟网络失败')
        s.agent.client=FailingClient()
        with patch.object(server.threading,'Thread',ImmediateThread):
            s.run_action('chat',{'text':'把顶部荷载改为12kN'})
        self.assertIsNone(s.agent.model)
        self.assertIsNone(s.agent.result)
        self.assertIsNone(s.agent.confirmed_hash)
        self.assertFalse(s.public()['can_confirm'])
        self.assertIsNone(s.public()['visual'])
        self.assertTrue(list((s.path/'history').glob('*.json')))

    def test_recovery_rejects_mismatching_model_hash(self):
        s=self.session();s.agent.record(recognition(cases()[0]))
        with patch.object(server.threading,'Thread',ImmediateThread):
            s.run_action('solve',{'model_hash':fingerprint(s.agent.model)})
        saved=json.loads((s.path/'session.json').read_text(encoding='utf-8'))
        saved['model']['materials'][0]['E']*=2
        (s.path/'session.json').write_text(json.dumps(saved),encoding='utf-8')
        restored=self.session()
        self.assertIsNone(restored.agent.model)
        self.assertIsNone(restored.agent.result)
        self.assertTrue(restored.error)


if __name__=='__main__':unittest.main()
