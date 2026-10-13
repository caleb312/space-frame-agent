"""保存记录改名、去重删除、刷新持久化及本地请求权限；仅操作临时数据。"""
from copy import deepcopy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from src.web import server
from src.model import parse_model
from src.solver import solve_structure
from tests.test_native_web import cases


class SavedResultsManagementTests(unittest.TestCase):
    def setUp(self):
        work = Path(__file__).resolve().parents[1]/'work'
        work.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='saved_management_',dir=work)
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)
        (self.root/'results').mkdir()
        for target,value in [('ROOT',self.root),('STORE',self.root/'results/web_sessions')]:
            context = patch.object(server,target,value)
            context.start();self.addCleanup(context.stop)
        context = patch.object(server,'defaults',return_value={'provider':'qwen','model':'test',
            'base_url':'https://example.invalid','vision':True,'thinking':'disabled',
            'timeout':1,'max_tokens':1024,'environment_key_available':False})
        context.start();self.addCleanup(context.stop)
        self.model = parse_model(cases()[0]).model_dump(mode='json')
        self.result = solve_structure(self.model).as_dict()
        self.save('a.json')

    def save(self,name,model=None):
        model = model or self.model
        result = self.result if model==self.model else solve_structure(model).as_dict()
        server.write_json(self.root/'results'/name,{'model':model,'result':result})

    def test_rename_persists_without_changing_model_or_numerics(self):
        self.save('b.json')
        before = (self.root/'results/a.json').read_bytes()
        server.manage_saved_result('a.json',title='  三角桁架验证 <节点30>  ')
        items = server.saved_results()
        self.assertEqual(len(items),1)
        self.assertEqual(items[0]['title'],'三角桁架验证 <节点30>')
        self.assertEqual(items[0]['name'],'a.json')
        self.assertEqual((self.root/'results/a.json').read_bytes(),before)
        # 去重代表文件改变，仍应按模型恢复用户名称。
        (self.root/'results/a.json').unlink()
        self.assertEqual(server.saved_results()[0]['title'],'三角桁架验证 <节点30>')

    def test_rename_rejects_invalid_titles(self):
        for title in ('','  ','line\nbreak','a'*101,None,42):
            with self.subTest(title=title),self.assertRaises(ValueError):
                server.manage_saved_result('a.json',title=title)
        self.assertFalse((self.root/'results/.saved_catalog.json').exists())

    def test_delete_moves_all_duplicates_and_known_companions(self):
        self.save('b.json')
        for name in ('a.recognition.json','a.summary.md','b.answer.md','b.3d.html'):
            (self.root/'results'/name).write_text('test companion',encoding='utf-8')
        other = deepcopy(self.model);other['materials'][0]['E']*=2
        self.save('other.json',other)
        unrelated = self.root/'results/notes.json'
        unrelated.write_text('{"notes":"retain"}',encoding='utf-8')
        originals = {p.name:p.read_bytes() for p in (self.root/'results').iterdir() if p.is_file()}
        server.manage_saved_result('a.json',delete=True)
        items = server.saved_results()
        self.assertEqual([item['name'] for item in items],['other.json'])
        trash = next((self.root/'results/.trash').iterdir())
        manifest = json.loads((trash/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(set(manifest['files']),{'a.json','b.json','a.recognition.json','a.summary.md','b.answer.md','b.3d.html'})
        for name in manifest['files']:
            self.assertEqual((trash/name).read_bytes(),originals[name])
            self.assertFalse((self.root/'results'/name).exists())
        self.assertEqual(unrelated.read_bytes(),originals['notes.json'])
        with self.assertRaises(ValueError):server.manage_saved_result('a.json',delete=True)

    def test_delete_preserves_open_result_and_does_not_reappear_on_archive(self):
        session = server.Session('1'*32,'qwen')
        session.agent.model = deepcopy(self.model)
        session.agent.result = deepcopy(self.result)
        session.agent.confirmed_hash = self.result['model_hash']
        session.add('notice','current result')
        server.manage_saved_result('a.json',delete=True)
        session.archive_current()
        self.assertEqual(session.agent.result,self.result)
        self.assertEqual(session.agent.confirmed_hash,self.result['model_hash'])
        self.assertEqual(server.saved_results(),[])
        self.assertFalse(list((self.root/'results').glob('web_*.json')))
        self.assertTrue(list((session.path/'history').glob('*.json')))

    def test_unlisted_paths_cannot_be_managed(self):
        for name in ('../a.json','..\\a.json','C:\\outside.json','.saved_catalog.json','missing.json'):
            with self.subTest(name=name),self.assertRaises(ValueError):
                server.manage_saved_result(name,delete=True)
        self.assertTrue((self.root/'results/a.json').exists())

    def test_failed_move_rolls_back_and_keeps_record_visible(self):
        self.save('b.json')
        original = Path.replace
        def fail_second(path,destination):
            if path.name=='b.json' and '.trash' in destination.parts:
                raise OSError('simulated move failure')
            return original(path,destination)
        with patch.object(Path,'replace',fail_second),self.assertRaises(OSError):
            server.manage_saved_result('a.json',delete=True)
        self.assertTrue((self.root/'results/a.json').exists())
        self.assertTrue((self.root/'results/b.json').exists())
        self.assertEqual(len(server.saved_results()),1)

    def test_open_renamed_result_reuses_saved_numerics(self):
        server.manage_saved_result('a.json',title='核对完成的桁架')
        session = server.Session('2'*32,'qwen')
        with patch('src.agent.runtime.solve_structure',side_effect=AssertionError('must not recalculate')):
            session.open_saved('a.json')
        self.assertEqual(session.agent.result,self.result)
        self.assertIn('核对完成的桁架',session.journal[0]['text'])

    def test_http_actions_require_session_token_and_reject_busy_session(self):
        local = server.LocalServer(0,'qwen')
        worker = threading.Thread(target=local.serve_forever,daemon=True)
        worker.start()
        def shutdown():
            local.shutdown();local.server_close();worker.join(timeout=2)
        self.addCleanup(shutdown)
        conn = http.client.HTTPConnection('127.0.0.1',local.server_port,timeout=3)
        self.addCleanup(conn.close)
        conn.request('GET','/api/state')
        response = conn.getresponse()
        cookie = response.getheader('Set-Cookie').split(';')[0]
        token = json.loads(response.read())['token']
        def post(action, supplied=token, **values):
            headers = {'Content-Type':'application/json','Cookie':cookie,'X-Frame-Token':supplied,
                'Origin':f'http://127.0.0.1:{local.server_port}'}
            conn.request('POST','/api/action',body=json.dumps({'action':action,'name':'a.json',**values}),headers=headers)
            response = conn.getresponse();status=response.status;response.read()
            return status
        self.assertEqual(post('delete_saved',supplied='wrong'),403)
        self.assertTrue((self.root/'results/a.json').exists())
        session = next(iter(local.sessions.values()));session.busy=True
        self.assertEqual(post('rename_saved',title='blocked'),400)
        session.busy=False
        self.assertEqual(post('rename_saved',title='HTTP名称'),200)
        self.assertEqual(server.saved_results()[0]['title'],'HTTP名称')
        self.assertEqual(post('delete_saved'),200)
        self.assertEqual(server.saved_results(),[])


if __name__=='__main__':unittest.main()
