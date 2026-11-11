"""命令行入口。

    # 合成数据 + 真实工具（需先运行 datagen.synth 和 models.train）
    python -m agent.run "预测 T03 未来一小时的功率" --as-of 2023-03-04T14:00
    # 假工具，不需要数据和模型
    python -m agent.run "预测 T03 未来一小时的功率，并解释它为什么比相邻机组低" --mock curtailment
"""
from __future__ import annotations

import argparse

from agent.mock_tools import SCENARIOS, mock_toolbox
from agent.report import render_report
from agent.state_machine import Request, run
from agent.toolbox import real_toolbox


def main() -> None:
    p = argparse.ArgumentParser(description="风电功率预测与异常复核智能体")
    p.add_argument("text", nargs="?", default="预测 T03 未来一小时的功率，并解释它为什么比相邻机组低")
    p.add_argument("--as-of", default="2019-07-14T14:00")
    p.add_argument("--mock", choices=SCENARIOS, help="使用假工具和指定场景；不加则用合成数据上的真实工具")
    args = p.parse_args()

    tools = mock_toolbox(args.mock) if args.mock else real_toolbox()
    print(render_report(run(Request(args.text, args.as_of), tools)))


if __name__ == "__main__":
    main()
