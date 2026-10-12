"""Standard-library-only Windows launcher; application and Solver stay separate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener
import venv
import webbrowser

ROOT = Path(__file__).resolve().parent
ENV = ROOT / '.venv'
LOCAL = ROOT / '.local'


def service_state(port):
    """Use loopback directly, even when the computer has an HTTP proxy."""
    url = f'http://127.0.0.1:{port}/'
    try:
        with build_opener(ProxyHandler({})).open(url, timeout=2) as response:
            return 'ours' if response.headers.get('Server', '').startswith('SpaceFrameLocal/') else 'other'
    except HTTPError:
        return 'other'
    except (URLError, TimeoutError, OSError):
        return 'unavailable'


def dependency_check(python):
    # Import each library as well as checking the version: installed metadata
    # alone does not prove that native NumPy/Pillow binaries can load.
    code = '''
import importlib, importlib.metadata, pathlib, re
modules = {'numpy': 'numpy', 'pydantic': 'pydantic', 'Pillow': 'PIL'}
def version(text):
    numbers = re.match(r'^(\\d+(?:\\.\\d+)*)', text).group(1)
    values = tuple(int(x) for x in numbers.split('.'))
    return (values + (0, 0, 0))[:3]
for line in pathlib.Path('requirements.txt').read_text(encoding='utf-8').splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    match = re.fullmatch(r'([A-Za-z0-9_-]+)((?:(?:>=|<)[0-9.]+,?)+)', line)
    if not match:
        raise ValueError('Unsupported dependency declaration: ' + line)
    name, constraints = match.groups()
    actual = version(importlib.metadata.version(name))
    for operator, bound in re.findall(r'(>=|<)([0-9.]+)', constraints):
        assert (actual >= version(bound)) if operator == '>=' else (actual < version(bound)), name
    importlib.import_module(modules[name])
'''
    return subprocess.run([str(python), '-c', code], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def run_logged(command):
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL / 'startup.log').open('a', encoding='utf-8') as log:
        log.write('\nPreparing local runtime\n')
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('运行环境准备失败，请检查网络连接。详细记录：.local/startup.log')


def prepare_runtime():
    python = ENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        print('首次启动：正在创建项目独立运行环境……', flush=True)
        venv.EnvBuilder(with_pip=True).create(ENV)
    if not dependency_check(python):
        print('正在安装所需依赖，首次准备需要联网，请稍候……', flush=True)
        run_logged([str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                    '--retries', '1', '--timeout', '30', '-r', str(ROOT / 'requirements.txt')])
        if not dependency_check(python):
            raise RuntimeError('依赖安装后仍无法加载，请查看 .local/startup.log。')
    LOCAL.mkdir(exist_ok=True)
    (LOCAL / 'runtime.json').write_text(json.dumps({
        'python': str(python),
        'requirements_sha256': hashlib.sha256((ROOT / 'requirements.txt').read_bytes()).hexdigest(),
    }, indent=2), encoding='utf-8')
    return python


def main():
    parser = argparse.ArgumentParser(description='一键启动空间结构工作台')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--provider', choices=['deepseek', 'qwen', 'gpt'], default='qwen')
    parser.add_argument('--check', action='store_true', help='只准备和检查运行环境')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('端口须为1024～65535')
    if sys.version_info < (3, 10):
        print('请安装 Python 3.10 或更新版本。', flush=True)
        return 1
    if not args.check:
        state = service_state(args.port)
        if state == 'ours':
            webbrowser.open(f'http://127.0.0.1:{args.port}/')
            print('工作台已经运行，已打开页面。', flush=True)
            return 0
        if state == 'other':
            print(f'端口 {args.port} 被其他程序占用。可运行：启动网页.cmd --port 8766', flush=True)
            return 1
    try:
        python = prepare_runtime()
        print('运行环境已准备好。', flush=True)
        if args.check:
            return 0
        print('正在打开工作台。演示期间请保留此窗口；关闭窗口或按 Ctrl+C 可停止服务。', flush=True)
        return subprocess.call([str(python), str(ROOT / 'web_main.py'), '--open',
                                '--port', str(args.port), '--provider', args.provider], cwd=ROOT)
    except KeyboardInterrupt:
        print('\n工作台已停止。', flush=True)
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
