"""独立材料/截面参数计算，无须 API 密钥。"""
import argparse
import json
from pathlib import Path

from pydantic import ValidationError
from src.solver import calculate_material_section
from src.solver.parameters import format_parameters


def main():
    parser = argparse.ArgumentParser(description='教材式材料与实心圆截面参数计算')
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/material_section_result.json'))
    args = parser.parse_args()
    try:
        result = calculate_material_section(json.loads(args.input.read_text(encoding='utf-8-sig')))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        print(format_parameters(result))
        print('计算记录：'+str(args.output.resolve()))
    except (OSError, ValueError, ValidationError) as error:
        print('参数计算失败：'+str(error))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
