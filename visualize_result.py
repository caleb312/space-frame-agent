"""将已保存结果或完整模型生成离线三维 HTML，不进行求解或 API 调用。"""
import argparse
import json
from pathlib import Path
import webbrowser

from src.agent.explanation import load_saved_result
from src.visualization import write_frame_html


def main():
    parser=argparse.ArgumentParser(description='空间刚架结构与结果三维查看')
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--input',type=Path,default=None,help='Agent 保存的 model/result JSON，默认 results/agent_result.json')
    source.add_argument('--model',type=Path,help='只预览完整 StructuralModel JSON，不显示未求解结果')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--open',action='store_true',help='在默认浏览器打开生成的图')
    args=parser.parse_args()
    try:
        if args.model:
            data=json.loads(args.model.read_text(encoding='utf-8-sig'))
            result=None
            path=args.output or Path('results')/(args.model.stem+'.model3d.html')
        else:
            source=args.input or Path('results/agent_result.json')
            saved=json.loads(source.read_text(encoding='utf-8-sig'))
            data=saved['model']
            result=load_saved_result(source)
            path=args.output or source.with_name(source.stem+'.3d.html')
        output=write_frame_html(data,result,path)
        print('三维文件已生成：'+output['path'])
        print('模式：'+output['mode']+'；模型哈希：'+output['model_hash'])
        if args.open: webbrowser.open(Path(output['path']).as_uri())
    except (OSError,ValueError,KeyError) as error:
        print('三维生成失败：'+str(error))
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
