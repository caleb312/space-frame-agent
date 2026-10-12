"""独立图片模式的请求隔离、诊断与模型保留；不请求真实API。"""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from src.agent.providers import APIConfig, APIError, ChatClient
from src.agent.vision import read_structure_image
from src.web.server import Session

ROOT = Path(__file__).resolve().parents[1]


class VisionReadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(dir=ROOT/'tests',prefix='vision_tmp_')
        # Windows递归清理前确认目标位于本测试目录。
        self.assertEqual(Path(self.tmp.name).resolve().parent,(ROOT/'tests').resolve())
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.image = self.path/'input_test.png'
        Image.new('RGB',(32,32),'white').save(self.image)
        self.client = ChatClient(APIConfig('qwen','qwen3-vl-plus','test-key',vision=True))

    def test_real_adapter_payload_has_one_image_no_tools(self):
        requests = []
        def response(request, timeout):
            requests.append(json.loads(request.data))
            packet={'model':'qwen3-vl-plus','choices':[{'index':0,
                'finish_reason':'stop','delta':{'role':'assistant','content':'①：未知'}}]}
            return BytesIO(('data: '+json.dumps(packet)+'\n\ndata: [DONE]\n\n').encode())
        with patch('src.agent.providers.urlopen',side_effect=response):
            reading = read_structure_image(self.client,self.image,'仅识别端点')
        payload = requests[0]
        self.assertEqual(len(payload['messages']),2)
        self.assertLess(len(payload['messages'][0]['content']),1000)
        self.assertEqual(payload['messages'][1]['content'][1]['type'],'image_url')
        self.assertNotIn('tools',payload)
        self.assertEqual(payload['max_tokens'],4096)
        self.assertEqual(reading['request']['image_count'],1)
        self.assertEqual(reading['request']['returned_model'],'qwen3-vl-plus')
        self.assertEqual(reading['request']['status'],'received')
        self.assertNotIn('test-key',json.dumps(reading))

    def test_tool_call_response_is_rejected(self):
        self.client.complete = lambda *args: {'content':'','tool_calls':[{'id':'fake'}]}
        with self.assertRaises(APIError):
            read_structure_image(self.client,self.image,'读图')

    def test_no_vision_is_rejected_before_api(self):
        self.client.config.vision = False
        with patch.object(self.client,'complete',side_effect=AssertionError('不可调用')):
            with self.assertRaises(ValueError):
                read_structure_image(self.client,self.image,'读图')

    def test_worker_keeps_model_confirmation_result_and_agent_history(self):
        with patch('src.web.server.STORE',self.path/'sessions'):
            session = Session('a'*32,'qwen')
            saved = json.loads((ROOT/'tests/fixtures/iv2_display_saved.json').read_text(encoding='utf-8'))
            session.agent.model = saved['model']
            session.agent.result = saved['result']
            session.agent.confirmed_hash = saved['result']['model_hash']
            session.agent.client = self.client
            session.agent.messages.append({'role':'user','content':'旧错误记录不能带入'})
            prior = deepcopy((session.agent.model,session.agent.result,session.agent.confirmed_hash,
                              session.agent.messages,session.agent.image_info))
            destination = session.path/'input_test.png'
            destination.write_bytes(self.image.read_bytes())
            captured = []
            def complete(messages, tools, purpose):
                captured.append(messages)
                return {'content':'仅识读回复'}
            self.client.complete = complete
            session.worker('read_image',{},'读图','input_test.png')
            self.assertFalse(session.error)
            self.assertEqual(session.reading['text'],'仅识读回复')
            self.assertNotIn('旧错误记录不能带入',json.dumps(captured))
            self.assertEqual((session.agent.model,session.agent.result,session.agent.confirmed_hash,
                              session.agent.messages,session.agent.image_info),prior)
            data = json.loads((session.path/'session.json').read_text(encoding='utf-8'))
            self.assertEqual(data['reading']['status'],'unverified')
            self.assertNotIn('test-key',json.dumps(data))
            self.assertIsNotNone(session.public()['reading'])

    def test_action_requires_new_image(self):
        with patch('src.web.server.STORE',self.path/'sessions'):
            session = Session('b'*32,'qwen')
            session.agent.client = self.client
            with self.assertRaisesRegex(ValueError,'随本轮上传图片'):
                session.run_action('read_image',{'text':'读图'})
            self.assertFalse(session.busy)


if __name__ == '__main__':
    unittest.main()
