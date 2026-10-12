"""凭据查询模型目录；不调用推理，不返回或保存密钥。"""
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlencode
from urllib.request import Request, urlopen
from .providers import APIError


def list_models(config):
    base = config.base_url.rstrip('/')
    started = time.monotonic()
    records = []
    for page in range(1, 11):
        if config.provider == 'qwen':
            parsed = urlparse(base)
            if parsed.path.rstrip('/') != '/compatible-mode/v1':
                raise APIError('Qwen 自动查询需要标准 compatible-mode/v1 地址；可手动填写模型名')
            url = parsed.scheme+'://'+parsed.netloc+'/api/v1/models?'+urlencode({'page_no':page,'page_size':100,'providers':'qwen'})
        else:
            url = base+'/models'
        remaining = 20-(time.monotonic()-started)
        if remaining <= 0:
            raise APIError('模型列表查询超过20秒，请稍后重试或手动填写模型名')
        request = Request(url, headers={'Authorization':'Bearer '+config.api_key,'Accept':'application/json'})
        try:
            with urlopen(request,timeout=remaining) as response:
                content = response.read(4*1024*1024+1)
            if len(content)>4*1024*1024:
                raise APIError('模型目录超出本地接收上限')
            data = json.loads(content)
        except HTTPError as error:
            raise APIError(f'模型查询 HTTP {error.code}：请检查密钥、地域和目录接口权限；可手动选择模型') from None
        except (URLError, OSError):
            raise APIError('模型列表网络连接失败或超时；已有连接未改变，可稍后重试') from None
        except ValueError:
            raise APIError('模型目录返回格式错误') from None
        output = data.get('output') or {}
        items = output.get('models') if config.provider=='qwen' else data.get('data')
        if not isinstance(items,list):
            raise APIError('此接口未提供模型目录，可手动填写模型名')
        for item in items:
            name = item.get('model') if config.provider=='qwen' else item.get('id')
            if not isinstance(name,str) or not name or len(name)>180:
                continue
            modalities = (item.get('inference_metadata') or {}).get('request_modality')
            capabilities = item.get('capabilities') or []
            features = item.get('features')
            vision = ('Image' in modalities) if isinstance(modalities,list) else ('VU' in capabilities) if capabilities else None
            tools = ('function-calling' in features) if isinstance(features,list) else None
            records.append({'id':name,'vision':vision,'tools':tools})
        if config.provider!='qwen' or len(records)>=output.get('total',len(records)) or not items:
            break
    unique = {item['id']:item for item in records}
    return sorted(unique.values(),key=lambda item:(item['vision'] is not True,item['id']))
