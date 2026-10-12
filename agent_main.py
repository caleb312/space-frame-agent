"""三家 API 的交互演示入口。运行方式见 docs/agent.md。"""
import argparse
import json
from pathlib import Path
import webbrowser

from src.agent import APIConfig, APIError, ChatClient, FrameAgent
from src.agent.runtime import fingerprint
from src.agent.explanation import result_table


def save_recognition(agent, output):
    if agent.draft is None:
        return
    path = output.with_name(output.stem+'.recognition.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'provider': agent.client.config.provider,
                               'model_name': agent.client.config.model,
                               'image_input': agent.image_info,
                               'draft': agent.draft, 'events': agent.events},
                              ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('识别记录、来源与阻塞项已保存：'+str(path.resolve()))


def main():
    parser = argparse.ArgumentParser(description='空间刚架 Agent：文字/图片 → 模型确认 → Solver')
    parser.add_argument('--provider', required=True, choices=['deepseek', 'qwen', 'gpt'])
    parser.add_argument('--image', type=Path, help='首轮本地图片；内容将发送到所选 API')
    parser.add_argument('--output', type=Path, default=Path('results/agent_result.json'))
    args = parser.parse_args()
    try:
        agent = FrameAgent(ChatClient(APIConfig.from_env(args.provider)))
        agent.visualization_output=str(args.output.with_name(args.output.stem+'.3d.html'))
        print('命令：/model 查看模型，/view 三维查看，/confirm 确认并求解，/explain 解释结果，/paste 多行输入，/quit 退出。')
        image = args.image
        if image:
            print('首轮图片文件：'+str(image.resolve()))
        while True:
            text = input('你：').strip()
            if text == '/quit':
                break
            if text == '/model':
                print(json.dumps(agent.draft, ensure_ascii=False, indent=2))
                continue
            if text == '/paste':
                print('粘贴多行内容，最后单独输入 /end；内容合并为一次模型请求。')
                lines=[]
                while True:
                    line=input()
                    if line.strip()=='/end':
                        break
                    lines.append(line)
                text='\n'.join(lines).strip()
            try:
                if text == '/view':
                    output=agent.visualize('results' if agent.result is not None else 'structure')
                    if output['status']=='visualized':
                        print('三维文件：'+output['path'])
                        webbrowser.open(Path(output['path']).as_uri())
                    else:
                        print('三维查看未完成：'+output['message'])
                    continue
                if text == '/explain':
                    if agent.result is None:
                        print('当前会话没有求解结果；已保存文件可用 explain_result.py 单独解释。')
                        continue
                    print(f'正在生成文字解释，最多等待 {agent.client.config.explain_timeout:g} 秒……',flush=True)
                    answer=agent.explain()
                    print('Agent：'+answer)
                    args.output.with_name(args.output.stem+'.answer.md').write_text(answer,encoding='utf-8')
                    continue
                if text == '/confirm':
                    if agent.model is None:
                        print(agent.draft['summary'] if agent.draft else '尚无有效识别记录，请先让 Agent 识别并提交记录。')
                        continue
                    agent.confirm(fingerprint(agent.model))
                    # 用户确认后先由程序调用 Solver，确保计算不会被文字响应跳过。
                    output = agent.solve()
                    agent.events.append({'tool': 'solve_space_frame', 'status': output['status']})
                    if agent.result:
                        # 即使解释 API 随后超时，也保留实际模型与计算证据。
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps({'model': agent.model, 'result': agent.result,
                                                           'events': agent.events}, ensure_ascii=False, indent=2), encoding='utf-8')
                        print('原始计算结果已保存：'+str(args.output.resolve()))
                        visual=agent.visualize('results')
                        if visual['status']=='visualized':
                            print('三维结果已保存：'+visual['path']+'；输入 /view 可打开')
                        else:
                            print('三维生成未完成：'+visual['message'])
                        table=result_table(agent.result)
                        args.output.with_name(args.output.stem+'.summary.md').write_text(table,encoding='utf-8')
                        print(table,flush=True)
                        print(f'正在生成文字解释，最多等待 {agent.client.config.explain_timeout:g} 秒……',flush=True)
                        try:
                            answer=agent.explain()
                            args.output.with_name(args.output.stem+'.answer.md').write_text(answer,encoding='utf-8')
                            print('Agent：'+answer,flush=True)
                        except APIError as error:
                            print('计算成功，文字解释未完成：'+str(error)+'；程序结果表已保存。',flush=True)
                    else:
                        print('结构求解未完成：'+json.dumps(output,ensure_ascii=False))
                elif text:
                    old_count = len(agent.messages)
                    try:
                        print('Agent：'+agent.chat(text, image))
                    finally:
                        # 失败后图片仍留在历史中，下一条信息不要再次追加同一张图。
                        if len(agent.messages) > old_count:
                            image = None
                else:
                    continue
                save_recognition(agent, args.output)
                if agent.result and text != '/confirm':
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps({'model': agent.model, 'result': agent.result,
                                                       'events': agent.events}, ensure_ascii=False, indent=2), encoding='utf-8')
                    print('原始计算结果已保存：'+str(args.output.resolve()))
            except (APIError, ValueError, OSError) as error:
                print('本轮未完成：'+str(error))
    except (APIError, ValueError, OSError) as error:
        print('无法启动：'+str(error))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
