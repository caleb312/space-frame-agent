"""杆系结构教学主程序：输入 → 按类型分派 → 直接刚度法 → 输出。"""
import argparse
import json
from pathlib import Path
import sys

from pydantic import ValidationError

from src.model import parse_model
from src.solver import SolverError, solve_structure


def main():
    parser = argparse.ArgumentParser(description='线弹性平面/空间桁架与刚架计算；课件顺时针转角/力矩为正')
    parser.add_argument('model', type=Path, help='结构模型 JSON 文件')
    parser.add_argument('--output', type=Path, help='结果 JSON 文件')
    parser.add_argument('--diagnostics', action='store_true', help='输出课件数组、单刚、总刚和 LDLT 分解')
    args = parser.parse_args()
    try:
        # 读取结构数字化文件，并校验节点、连接、材料、截面与局部轴。
        model = parse_model(json.loads(args.model.read_text(encoding='utf-8-sig')))
        # 内部主流程保留课件子函数职责，可独立调用和测试。
        result = solve_structure(model)
        # 数值、单位、外部编号与模型哈希同时输出，便于 Agent 追溯。
        text = json.dumps(result.as_dict(args.diagnostics), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text+'\n', encoding='utf-8')
            print(f'计算完成：{args.output}')
        else:
            print(text)
    except (OSError, ValidationError, ValueError, SolverError) as error:
        print(f'计算失败：{error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
