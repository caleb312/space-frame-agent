"""课程演示用三家 API 适配；不依赖 Agent 框架或供应商 SDK。"""
import base64
from dataclasses import dataclass, field
import hashlib
from http.client import RemoteDisconnected, IncompleteRead
from io import BytesIO
import json
import math
import os
from pathlib import Path
import socket
import ssl
import time
from queue import Queue, Empty
from threading import Thread
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class APIError(RuntimeError):
    pass


PROVIDERS = {
    'deepseek': ('DEEPSEEK', 'https://api.deepseek.com'),
    'qwen': ('QWEN', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
    'gpt': ('OPENAI', 'https://api.openai.com/v1'),
}


def provider_environment(provider):
    """GPT 可读取已获用户批准的项目 .env.local；环境变量优先，不输出秘密。"""
    prefix = PROVIDERS[provider][0]
    names = [prefix+'_'+suffix for suffix in
             ('API_KEY', 'MODEL', 'BASE_URL', 'VISION', 'TIMEOUT', 'EXPLAIN_TIMEOUT')]
    values = {}
    if provider == 'gpt':
        path = Path(__file__).resolve().parents[2]/'.env.local'
        if path.is_file() and not path.is_symlink():
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                name, separator, value = line.strip().partition('=')
                if separator and name in names:
                    value = value.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                        value = value[1:-1]
                    values[name] = value
    for name in names:
        if name in os.environ:
            values[name] = os.environ[name]
    return values


def provider_key(provider, base_url):
    """项目保存的 OpenAI 密钥只允许发送到官方地址。"""
    prefix, official_base = PROVIDERS[provider]
    if os.getenv(prefix+'_API_KEY'):
        return os.environ[prefix+'_API_KEY']
    key = provider_environment(provider).get(prefix+'_API_KEY', '')
    if key and provider == 'gpt' and base_url.rstrip('/') != official_base:
        raise ValueError('项目保存的 OpenAI 密钥仅可用于 https://api.openai.com/v1')
    return key


@dataclass
class APIConfig:
    provider: str
    model: str
    api_key: str = field(repr=False)
    base_url: str = ''
    vision: bool = False
    timeout: float = 180
    deepseek_max_tokens: int = 32768
    deepseek_thinking: str = 'enabled'
    deepseek_reasoning_effort: str = 'low'
    explain_timeout: float = 60

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError('只支持 deepseek、qwen、gpt')
        if not self.model.strip() or not self.api_key.strip():
            raise ValueError('需要配置模型名称与 API 密钥')
        self.base_url = (self.base_url or PROVIDERS[self.provider][1]).rstrip('/')
        parsed = urlparse(self.base_url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('API 地址必须是无账号、查询参数和片段的 HTTPS 地址')
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('timeout 必须大于零')
        if not math.isfinite(self.explain_timeout) or self.explain_timeout <= 0:
            raise ValueError('解释等待时间必须为有限正数')
        if type(self.deepseek_max_tokens) is not int or not 1 <= self.deepseek_max_tokens <= 393216:
            raise ValueError('DEEPSEEK_MAX_TOKENS 必须为 1～393216 的整数')
        if self.deepseek_thinking not in ('enabled', 'disabled'):
            raise ValueError('DEEPSEEK_THINKING 只能为 enabled 或 disabled')
        if self.deepseek_reasoning_effort not in ('low', 'high', 'max'):
            raise ValueError('DEEPSEEK_REASONING_EFFORT 只能为 low、high 或 max')

    @classmethod
    def from_env(cls, provider):
        if provider not in PROVIDERS:
            raise ValueError('只支持 deepseek、qwen、gpt')
        prefix, base = PROVIDERS[provider]
        values = provider_environment(provider)
        base = values.get(prefix+'_BASE_URL', base)
        return cls(provider, values.get(prefix+'_MODEL', 'gpt-4.1' if provider == 'gpt' else ''),
                   provider_key(provider, base), base,
                   values.get(prefix+'_VISION', '1' if provider == 'gpt' else '0') == '1',
                   float(values.get(prefix+'_TIMEOUT', '180')),
                   deepseek_max_tokens=int(os.getenv('DEEPSEEK_MAX_TOKENS', '32768')) if provider == 'deepseek' else 32768,
                   deepseek_thinking=os.getenv('DEEPSEEK_THINKING', 'enabled') if provider == 'deepseek' else 'enabled',
                   deepseek_reasoning_effort=os.getenv('DEEPSEEK_REASONING_EFFORT', 'low') if provider == 'deepseek' else 'low',
                   explain_timeout=float(values.get(prefix+'_EXPLAIN_TIMEOUT','60')))


class ChatClient:
    def __init__(self, config):
        self.config = config

    def complete(self, messages, tools=None, purpose='agent'):
        if purpose != 'explanation':
            return self._complete(messages, tools, purpose)
        # HTTP 保活字节可能延长普通 socket 超时；解释阶段另设总等待上限。
        previous = getattr(self, '_explanation_thread', None)
        if previous is not None and previous.is_alive():
            raise APIError('上一条解释请求尚未结束，请稍后重试；结果表可直接查看')
        output = Queue(maxsize=1)
        def request():
            try:
                output.put((True, self._complete(messages, None, purpose)))
            except Exception as error:
                output.put((False, error))
        self._explanation_thread = Thread(target=request, daemon=True)
        self._explanation_thread.start()
        try:
            success, value = output.get(timeout=self.config.explain_timeout)
        except Empty:
            raise APIError(f'文字解释超过总等待上限 {self.config.explain_timeout:g} 秒；计算结果已保留') from None
        if not success:
            raise value
        return value

    def _complete(self, messages, tools, purpose):
        payload = {'model': self.config.model, 'messages': messages, 'stream': False}
        if self.config.provider == 'qwen':
            payload['stream'] = True
            payload['stream_options'] = {'include_usage': True}
            # 快速识读/解释直接回答；完整建模保留有限推理。
            if self.config.model.startswith('qwen3.8-') and purpose == 'agent':
                payload['reasoning_effort'] = 'low'
            if purpose in ('vision', 'explanation') and self.config.model.startswith(('qwen3', 'qwen-plus', 'qwen-flash')):
                payload['enable_thinking'] = False
                payload['preserve_thinking'] = False
                if self.config.model.startswith('qwen3.8-'):
                    payload['reasoning_effort'] = 'none'
                payload['max_tokens'] = 4096 if purpose == 'vision' else 768
        if self.config.provider == 'deepseek':
            payload['max_tokens'] = self.config.deepseek_max_tokens
            payload['thinking'] = {'type': self.config.deepseek_thinking}
            if self.config.deepseek_thinking == 'enabled':
                payload['reasoning_effort'] = self.config.deepseek_reasoning_effort
            if purpose == 'explanation':
                payload['thinking'] = {'type':'disabled'}
                payload.pop('reasoning_effort', None)
                payload['max_tokens'] = 4096
        if purpose == 'vision':
            payload['max_tokens'] = 4096
            if self.config.provider == 'deepseek':
                payload['thinking'] = {'type':'disabled'}
                payload.pop('reasoning_effort', None)
        if self.config.provider == 'gpt':
            payload['max_completion_tokens'] = 4096 if purpose in ('vision', 'explanation') else 16384
            payload.pop('max_tokens', None)
        if tools:
            payload.update(tools=tools, tool_choice='auto')
        # 只记录诊断元数据；不记录密钥、原始图片、消息正文或思考内容。
        self.last_exchange = {'status':'requesting','requested_model':self.config.model,
            'purpose':purpose,'image_count':sum(
                1 for m in messages if isinstance(m.get('content'),list)
                for block in m['content'] if isinstance(block,dict) and block.get('type')=='image_url'),
            'tools_count':len(tools or []),'stream':payload['stream']}
        started = time.monotonic()
        request = Request(self.config.base_url+'/chat/completions',
                          data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8'),
                          headers={'Authorization': 'Bearer '+self.config.api_key,
                                   'Content-Type': 'application/json'}, method='POST')
        try:
            timeout = self.config.explain_timeout if purpose == 'explanation' else self.config.timeout
            with urlopen(request, timeout=timeout) as response:
                data = self._read_stream(response, started, timeout) if payload['stream'] else json.load(response)
            choice = data['choices'][0]
            finish = choice.get('finish_reason')
            raw = choice['message']
            if raw.get('role') != 'assistant':
                raise APIError('API 返回了无效的消息角色')
            self.last_exchange.update(returned_model=data.get('model') if isinstance(data.get('model'),str) else None,
                                      finish_reason=finish)
            if finish == 'length':
                usage = data.get('usage') or {}
                tokens = usage.get('completion_tokens', '未返回')
                # 独立识图只有文字，不进入模型。允许展示已收到的截断文字，但显式标记不完整。
                # 结构建模/解释仍拒绝截断输出，尤其不能执行不完整工具参数。
                if purpose == 'vision' and not tools and not raw.get('tool_calls') and isinstance(raw.get('content'),str) and raw['content'].strip():
                    self.last_exchange.update(status='received',incomplete=True,completion_tokens=tokens)
                    return {'role':'assistant','content':raw['content']}
                hint = ('DeepSeek 可调整 MAX_TOKENS 或 THINKING 配置。' if self.config.provider=='deepseek'
                        else '请缩小本轮任务范围；结构建模不能使用截断参数。')
                raise APIError(f'模型输出达到长度上限（finish_reason=length，输出 tokens={tokens}）；'
                               '本轮未执行工具。'+hint)
            if finish == 'content_filter':
                raise APIError('模型服务过滤了本轮内容（finish_reason=content_filter）；本轮未执行工具')
            if finish in ('insufficient_system_resource', 'aborted'):
                raise APIError(f'模型服务中断了生成（finish_reason={finish}）；本轮未执行工具，请稍后重试')
            self.last_exchange.update(status='received',
                returned_model=data.get('model') if isinstance(data.get('model'),str) else None,
                finish_reason=finish)
            # DeepSeek 思考模式工具续调用需要原 reasoning_content；不打印此字段。
            return {k: v for k, v in raw.items()
                    if k in ('role', 'content', 'tool_calls', 'reasoning_content')}
        except HTTPError as error:
            # 仅提取白名单错误码；不打印原始服务响应、消息或请求。
            service_code = None
            try:
                failure = json.loads(error.read(65536))
                service_code = (failure.get('error') or {}).get('code') or failure.get('code')
            except (ValueError, TypeError, AttributeError, OSError):
                pass
            if not isinstance(service_code, str):
                service_code = None
            known = {'Arrearage':'账号欠费或余额不足，请检查百炼账单',
                     'insufficient_quota':'服务返回额度限制，请检查账号额度及调用限流',
                     'Throttling.AllocationQuota':'Token 调用限流，请稍后再试',
                     'Throttling.RateQuota':'请求频率超限，请稍后再试',
                     'rate_limit_exceeded':'请求限流，请稍后再试',
                     'invalid_api_key':'密钥无效，请重新配置',
                     'model_not_found':'模型不存在或没有调用权限'}
            note = known.get(service_code, '请检查密钥、模型权限、额度与地址')
            self.last_exchange.update(status='failed', http_status=error.code,
                error_code=service_code if service_code in known else None,
                elapsed_seconds=round(time.monotonic()-started,1))
            raise APIError(f'API HTTP {error.code}：{note}') from None
        except (RemoteDisconnected, IncompleteRead) as error:
            self.last_exchange.update(status='failed', error_type=type(error).__name__,
                elapsed_seconds=round(time.monotonic()-started,1))
            raise APIError('API 连接被远端或中间网络关闭，未收到完整回复；本轮未完整接收的工具不会执行。'
                           '未取得额度错误，不能据此判断余额不足。') from None
        except APIError:
            self.last_exchange.update(status='failed', elapsed_seconds=round(time.monotonic()-started,1))
            raise
        except (URLError, TimeoutError, OSError) as error:
            reason = error.reason if isinstance(error, URLError) else error
            if isinstance(reason, (TimeoutError, socket.timeout)):
                option = 'EXPLAIN_TIMEOUT' if purpose == 'explanation' else 'TIMEOUT'
                message = f'API 请求超时（{timeout:g} 秒）；可增加对应服务商的 {option} 环境变量后重启'
            elif isinstance(reason, ssl.SSLCertVerificationError):
                message = 'HTTPS 证书校验失败；请检查系统时间和代理证书配置'
            elif isinstance(reason, ssl.SSLError):
                message = 'HTTPS/TLS 连接失败；请检查代理或网络连接'
            elif isinstance(reason, socket.gaierror):
                message = 'API 域名解析失败；请检查 DNS 和接口地址'
            elif isinstance(reason, PermissionError):
                message = '网络连接被系统权限或防火墙阻止'
            else:
                code = getattr(reason, 'winerror', None) or getattr(reason, 'errno', None)
                message = f'API 网络连接失败（{type(reason).__name__}，错误码 {code}）；请检查代理和网络'
            self.last_exchange.update(status='failed', error_type=type(reason).__name__,
                elapsed_seconds=round(time.monotonic()-started,1))
            raise APIError(message) from None
        except (KeyError, IndexError, TypeError, ValueError):
            self.last_exchange.update(status='failed', error_type='invalid_response',
                elapsed_seconds=round(time.monotonic()-started,1))
            raise APIError('API 返回格式不符合 Chat Completions 协议') from None

    def _read_stream(self, response, started, timeout):
        """合并 Qwen SSE 文本、思考与工具参数；完整结束前不向 Agent 返回工具。"""
        raw = {'role':'assistant', 'content':''}
        calls = {}
        finish = None
        model = None
        usage = {}
        event_lines = []
        finished = False
        received_bytes = 0
        self.last_exchange.update(status='streaming', chunks_received=0)

        def consume(event):
            nonlocal finish, model, usage, finished
            if event == '[DONE]':
                finished = True
                return
            data = json.loads(event)
            if data.get('error'):
                raise APIError('API 在流式生成中返回错误；回复未完整接收，工具未执行')
            self.last_exchange['chunks_received'] += 1
            self.last_exchange['elapsed_seconds'] = round(time.monotonic()-started,1)
            if isinstance(data.get('model'), str):
                model = data['model']
            if isinstance(data.get('usage'), dict):
                usage = data['usage']
            for choice in data.get('choices', []):
                if choice.get('index', 0) != 0:
                    continue
                delta = choice.get('delta') or {}
                if delta.get('role') not in (None, 'assistant'):
                    raise ValueError('Unexpected stream role')
                for key in ('content', 'reasoning_content'):
                    fragment = delta.get(key)
                    if fragment is not None:
                        if not isinstance(fragment, str):
                            raise ValueError('Invalid stream text')
                        raw[key] = raw.get(key, '') + fragment
                for part in delta.get('tool_calls') or []:
                    index = part.get('index')
                    if type(index) is not int or not 0 <= index < 128:
                        raise ValueError('Invalid tool index')
                    call = calls.setdefault(index, {'id':'','type':'function',
                        'function':{'name':'','arguments':''}})
                    if part.get('id'):
                        if call['id'] and call['id'] != part['id']:
                            raise ValueError('Conflicting tool id')
                        call['id'] = part['id']
                    if part.get('type') not in (None, 'function'):
                        raise ValueError('Invalid tool type')
                    function = part.get('function') or {}
                    for key in ('name', 'arguments'):
                        fragment = function.get(key)
                        if fragment is not None:
                            if not isinstance(fragment, str):
                                raise ValueError('Invalid tool fragment')
                            call['function'][key] += fragment
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']

        for line in response:
            received_bytes += len(line)
            if received_bytes > 16*1024*1024:
                raise APIError('API 流式回复超过本地接收上限；未执行本次工具')
            if time.monotonic()-started > timeout:
                raise APIError(f'API 流式生成超过 {timeout:g} 秒；未完整接收的工具不会执行')
            line = line.decode('utf-8').rstrip('\r\n')
            if not line:
                if event_lines:
                    consume('\n'.join(event_lines))
                    event_lines.clear()
                    if finished:
                        break
            elif line.startswith('data:'):
                event_lines.append(line[5:].lstrip(' '))
        if event_lines and not finished:
            consume('\n'.join(event_lines))
        if not finished or finish is None:
            raise APIError('API 流式回复中途结束，未收到完整结束标记；本次工具未执行')
        if calls:
            raw['tool_calls'] = [calls[index] for index in sorted(calls)]
            if finish in ('stop', 'tool_calls'):
                for call in raw['tool_calls']:
                    if not call['id'] or not call['function']['name']:
                        raise ValueError('Incomplete tool call')
                    json.loads(call['function']['arguments'])
        return {'model':model, 'usage':usage,
                'choices':[{'message':raw,'finish_reason':finish}]}


def prepare_image(image_path):
    """API 上传预处理：落实 EXIF 显示方向，重新编码首帧；原文件不改动。"""
    from PIL import Image, ImageOps, UnidentifiedImageError

    path = Path(image_path)
    if path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp') or path.stat().st_size > 10*1024*1024:
        raise ValueError('图片须为 PNG/JPEG/WebP，且不超过 10 MiB')
    original = path.read_bytes()
    try:
        with Image.open(BytesIO(original)) as image:
            if image.format not in ('JPEG', 'MPO', 'PNG', 'WEBP'):
                raise ValueError('图片实际格式不是支持的 PNG/JPEG/WebP')
            stored_size = image.size
            orientation = image.getexif().get(274, 1)
            source_format = image.format
            normalized = ImageOps.exif_transpose(image)
            buffer = BytesIO()
            if source_format in ('JPEG', 'MPO'):
                normalized.convert('RGB').save(buffer, format='JPEG', quality=95, exif=b'')
                mime = 'image/jpeg'
            else:
                normalized.save(buffer, format='PNG', exif=b'')
                mime = 'image/png'
            content = buffer.getvalue()
            info = {'path': str(path.resolve()), 'stored_size': list(stored_size),
                    'uploaded_size': list(normalized.size), 'original_exif_orientation': orientation,
                    'orientation_applied': orientation in range(2, 9),
                    'original_sha256': hashlib.sha256(original).hexdigest(),
                    'uploaded_sha256': hashlib.sha256(content).hexdigest(), 'mime': mime}
    except (UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise ValueError('图片无法解码或像素尺寸超出限制') from error
    if len(content) > 10*1024*1024:
        raise ValueError('方向校正后的上传内容超过 10 MiB，请使用较小的 JPEG 图片')
    return content, info


def user_message(text, image_path=None, vision=False, image_info=None):
    if image_path is None:
        return {'role': 'user', 'content': text}
    if not vision:
        raise ValueError('当前配置未启用图片输入；请选择支持视觉与工具调用的模型并设置 VISION=1')
    content, info = prepare_image(image_path)
    if image_info is not None:
        image_info.clear()
        image_info.update(info)
    mime = info['mime']
    encoded = base64.b64encode(content).decode('ascii')
    return {'role': 'user', 'content': [
        {'type': 'text', 'text': text},
        {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{encoded}'}}]}
