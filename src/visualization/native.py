"""原生结构复用既有交互查看界面，数据由确定性后处理生成。"""
import hashlib
import html
import json
from pathlib import Path
import re
from .native_payload import build_native_payload


def write_native_html(data,result,path,title=None):
    payload=build_native_payload(data,result,title)
    here=Path(__file__).resolve().parent
    library=here/'assets/plotly-4.1.1.min.js'
    manifest=json.loads((here/'assets/manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(library.read_bytes()).hexdigest()!=manifest['sha256']:
        raise ValueError('Plotly资源哈希不一致')
    serialized=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<',chr(92)+'u003c')
    replacements={'TITLE':html.escape(payload['title']),'PLOTLY':library.read_text(encoding='utf-8'),'DATA':serialized}
    template=(here/'viewer.html').read_text(encoding='utf-8')
    output=re.sub(r'__(TITLE|PLOTLY|DATA)__',lambda match:replacements[match.group(1)],template)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(output,encoding='utf-8')
    return {'path':str(path.resolve()),'model_hash':payload['model_hash'],'mode':'results' if result is not None else 'structure'}
