"""启动个人课程项目的本地交互页面。"""
import argparse
import webbrowser
from urllib.request import urlopen
from src.web.server import LocalServer


def main():
    parser = argparse.ArgumentParser(description='空间刚架 Agent 本地网页')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--provider',choices=['deepseek','qwen','gpt'],default='deepseek')
    parser.add_argument('--open',action='store_true',help='启动后打开浏览器')
    args = parser.parse_args()
    if not 1024<=args.port<=65535:
        parser.error('端口须为1024～65535')
    try:
        server = LocalServer(args.port,args.provider)
    except OSError:
        if args.open:
            try:
                url = f'http://127.0.0.1:{args.port}'
                with urlopen(url,timeout=2) as response:
                    existing = response.headers.get('Server','').startswith('SpaceFrameLocal/')
                if existing:
                    webbrowser.open(url)
                    print('本地工作台已在运行，已打开页面。')
                    return
            except OSError:
                pass
        parser.error('端口已被占用或无法监听；可改用 --port 8766')
    url = f'http://127.0.0.1:{server.server_port}'
    print('空间刚架 Agent 本地页面：'+url,flush=True)
    print('API 在页面中配置；GPT 可读取已保存的 .env.local 密钥。Ctrl+C 停止。',flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n本地服务已停止，会话及计算结果保存在 results/web_sessions。')
    finally:
        server.server_close()


if __name__=='__main__':
    main()
