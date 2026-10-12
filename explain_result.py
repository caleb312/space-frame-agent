"""只解释已保存的计算结果；不重新识图、建模或结构求解。"""
import argparse
from pathlib import Path
from src.agent import APIConfig, APIError, ChatClient
from src.agent.explanation import load_saved_result, result_table, explain_result


def main():
    parser = argparse.ArgumentParser(description='查看结果表，并补生成已保存结果的文字解释')
    parser.add_argument('--provider', choices=['deepseek','qwen','gpt'], default='deepseek')
    parser.add_argument('--input', type=Path, default=Path('results/agent_result.json'))
    parser.add_argument('--local', action='store_true', help='仅生成程序结果表，不调用 API')
    args = parser.parse_args()
    try:
        result = load_saved_result(args.input)
        table = result_table(result)
        summary_path = args.input.with_name(args.input.stem+'.summary.md')
        summary_path.write_text(table, encoding='utf-8')
        print(table, flush=True)
        print('\n程序结果表已保存：'+str(summary_path.resolve()), flush=True)
        if args.local:
            return 0
        client = ChatClient(APIConfig.from_env(args.provider))
        print(f'正在生成文字解释，最多等待 {client.config.explain_timeout:g} 秒……',flush=True)
        try:
            answer = explain_result(client,result)
        except APIError as error:
            print('文字解释未完成：'+str(error)+'；上面的程序结果表仍可使用。',flush=True)
            return 1
        answer_path=args.input.with_name(args.input.stem+'.answer.md')
        answer_path.write_text(answer,encoding='utf-8')
        print('\nAgent：'+answer,flush=True)
        print('文字解释已保存：'+str(answer_path.resolve()))
    except (OSError,ValueError,KeyError,APIError) as error:
        print('无法读取或解释结果：'+str(error))
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
