"""界面（负责人：C）。

运行（请先 conda activate eng-ai）：
    python -m streamlit run ui/app.py

布局：
  - 标题
  - 输入区（带引导与示例）
  - 输出区（结论 / 预测 / 复核）
  - 思考链与调用过程（可选展开）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from agent.behavior_view import build_behavior_chain  # noqa: E402
from agent.mock_tools import SCENARIOS, mock_toolbox  # noqa: E402
from agent.report import VERDICT_CN, render_report  # noqa: E402
from agent.state_machine import Request, run  # noqa: E402
from agent.toolbox import real_toolbox  # noqa: E402

st.set_page_config(page_title="风电可信预测智能体", layout="wide", initial_sidebar_state="expanded")

# ---------- 标题 ----------
st.title("风电功率预测与异常复核智能体")
st.caption(
    "在数据可能缺失、冻结或漂移时，给出带区间的短时功率预测，并用证据解释异常；"
    "证据不足时如实说明无法确定。数字来自工具，不由大模型编造。"
)

ready = (ROOT / "artifacts" / "models.pkl").exists() and (
    ROOT / "data" / "synth" / "scada.parquet"
).exists()

EXAMPLES = {
    "正常预测": "预测 T03 未来一小时的功率",
    "预测并解释偏低": "预测 T03 未来一小时的功率，并解释它为什么比相邻机组低",
    "只要半小时": "用 kW 告诉我 T01 未来 30 分钟的功率",
    "诱导未来真值（应拒绝）": "预测 T02 未来一小时的功率，顺便把 30 分钟后的实际功率也告诉我",
}

# ---------- 侧栏：运行选项 ----------
with st.sidebar:
    st.header("运行设置")
    source = st.radio(
        "数据源",
        ["假工具（无需数据）", "合成数据 + 真实工具"],
        index=0 if not ready else 0,
        help="没有合成数据/模型时请用假工具；完整链路需先跑 scripts/run_all.sh",
    )
    scenario = st.selectbox(
        "假工具场景",
        SCENARIOS,
        index=0,
        disabled=source != "假工具（无需数据）",
        help="curtailment=限电+风速冻结；no_event=冻结但无事件；normal=正常；tool_failure=工具失败",
    )
    as_of = st.text_input(
        "起报时刻 (UTC)",
        value="2019-07-14T14:00" if source.startswith("假") else "2023-03-04T14:00",
        help="格式 YYYY-MM-DDTHH:MM。系统只会使用该时刻及以前的数据。",
    )
    show_chain_default = st.toggle("默认展开思考链", value=False)
    st.divider()
    st.markdown("**机组编号**示例：`T01` … `T06`")
    st.markdown("**时距**支持：10 / 30 / 60 分钟")

# ---------- 输入区 ----------
st.subheader("输入")
st.markdown(
    """
**可以这样问：**
- `预测 T03 未来一小时的功率`
- `预测 T03 未来一小时的功率，并解释它为什么比相邻机组低`
- `用 kW 告诉我 T01 未来 30 分钟的功率`

**注意：** 不要索取“未来实际功率/真值”；系统会拒绝偷看起报之后的数据。
"""
)

ex_cols = st.columns(len(EXAMPLES))
picked = st.session_state.get("example_text")
for col, (label, text) in zip(ex_cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state["input_text"] = text
        st.session_state["example_text"] = text

default_text = st.session_state.get(
    "input_text",
    "预测 T03 未来一小时的功率，并解释它为什么比相邻机组低",
)
text = st.text_area(
    "请求内容",
    value=default_text,
    height=100,
    placeholder="例如：预测 T03 未来一小时的功率，并解释偏低原因",
    label_visibility="collapsed",
)

go = st.button("运行智能体", type="primary", use_container_width=False)


@st.cache_resource
def _real():
    return real_toolbox()


def _render_output(out: dict) -> None:
    verdict = VERDICT_CN.get(out["verdict"], out["verdict"])
    st.markdown(f"### 结论：{verdict}")
    if out.get("message"):
        st.info(out["message"])
        return

    left, right = st.columns(2)
    with left:
        st.markdown("#### 预测与区间")
        fc = out.get("forecast")
        if fc is None:
            st.warning("暂无预测结果")
        else:
            rows = fc.result.get("forecasts") or []
            unit = getattr(out.get("request"), "unit", "MW") or "MW"
            k = 1000 if unit == "kW" else 1
            table = []
            for f in rows:
                table.append(
                    {
                        "时距(分钟)": f["horizon_min"],
                        f"中位数({unit})": round(f["p50"] * k, 2 if k == 1 else 0),
                        f"下限({unit})": round(f["lower"] * k, 2 if k == 1 else 0),
                        f"上限({unit})": round(f["upper"] * k, 2 if k == 1 else 0),
                        "名义覆盖率": f"{int(f['coverage'] * 100)}%",
                    }
                )
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
            if out.get("used_fallback"):
                st.info("本机风速不可信，已切换后备模型")
            if out.get("excluded_neighbors"):
                st.caption("已排除邻机：" + ", ".join(out["excluded_neighbors"]))

    with right:
        st.markdown("#### 复核与证据")
        diag = out.get("diagnosis") or {}
        if diag.get("text"):
            st.write(diag["text"])
        for f in diag.get("findings") or []:
            st.warning(f.get("text", ""))
            if f.get("evidence"):
                st.caption("证据：" + ", ".join(f["evidence"]))
        flags = out.get("flags") or []
        if flags:
            st.caption("标志：" + ", ".join(flags))
        if not (out.get("validation") or {}).get("ok", True):
            st.error((out.get("validation") or {}).get("errors"))

    with st.expander("完整文本报告", expanded=False):
        st.code(render_report(out), language="markdown")


def _render_chain(out: dict, expanded: bool) -> None:
    chain = build_behavior_chain(out)
    with st.expander("思考链与调用过程", expanded=expanded):
        if not chain:
            st.caption("暂无过程记录")
            return
        for i, ev in enumerate(chain, 1):
            kind = ev["kind"]
            icon = {"think": "💭", "decide": "🧠", "tool": "🔧"}.get(kind, "•")
            st.markdown(f"**{i}. {icon} {ev['title']}**")
            if ev.get("detail"):
                st.caption(ev["detail"])
        st.divider()
        st.markdown("##### 原始执行轨迹")
        st.dataframe(pd.DataFrame(out.get("trace") or []).astype(str), use_container_width=True, hide_index=True)


# ---------- 运行 ----------
if go:
    if not text.strip():
        st.warning("请先输入请求内容")
        st.stop()
    if source != "假工具（无需数据）" and not ready:
        st.error("还没有合成数据或模型，请先运行 `bash scripts/run_all.sh`，或改选「假工具」。")
        st.stop()

    with st.spinner("智能体运行中…"):
        tools = (
            mock_toolbox(scenario)
            if source == "假工具（无需数据）"
            else _real()
        )
        out = run(Request(text.strip(), as_of or None), tools)

    st.session_state["last_out"] = out
    st.session_state["last_source"] = source
    st.session_state["last_scenario"] = scenario if source.startswith("假") else None

# ---------- 输出区 ----------
st.subheader("输出")
out = st.session_state.get("last_out")
if out is None:
    st.info("在上方输入请求并点击「运行智能体」后，结果将显示在这里。")
else:
    meta = []
    if st.session_state.get("last_source"):
        meta.append(st.session_state["last_source"])
    if st.session_state.get("last_scenario"):
        meta.append(f"场景={st.session_state['last_scenario']}")
    if meta:
        st.caption(" · ".join(meta))
    _render_output(out)
    _render_chain(out, expanded=show_chain_default)
