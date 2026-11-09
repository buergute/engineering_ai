"""命令行入口。

    python -m agent.run --turbine T03 --as-of 2019-07-14T14:00 --mock curtailment
"""
from __future__ import annotations

import argparse

from agent.mock_tools import SCENARIOS, mock_toolbox
from agent.report import render_report
from agent.state_machine import Request, run
from agent.toolbox import real_toolbox


def main() -> None:
    p = argparse.ArgumentParser(description="风电功率预测与异常复核智能体")
    p.add_argument("--turbine", default="T03")
    p.add_argument("--as-of", default="2019-07-14T14:00")
    p.add_argument("--mock", choices=SCENARIOS, help="使用假工具和指定场景；不加则用真实工具")
    args = p.parse_args()

    tools = mock_toolbox(args.mock) if args.mock else real_toolbox()
    out = run(Request(turbine=args.turbine, as_of=args.as_of), tools)
    print(render_report(out))


if __name__ == "__main__":
    main()
