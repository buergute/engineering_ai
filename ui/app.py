"""界面（负责人：C，截止 11/18）。运行：streamlit run ui/app.py

四块：任务与起报时间 / 预测与区间 / 质量与证据 / 执行轨迹。目前接假工具。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from agent.mock_tools import SCENARIOS, mock_toolbox  # noqa: E402
from agent.state_machine import Request, run  # noqa: E402

st.set_page_config(page_title="风电预测与复核智能体", layout="wide")
st.title("风电功率预测与异常复核智能体")

with st.sidebar:
    st.header("任务与起报时间")
    turbine = st.text_input("机组", "T03")
    as_of = st.text_input("起报时刻", "2019-07-14T14:00")
    scenario = st.selectbox("假工具场景（开发用）", SCENARIOS)
    go = st.button("运行")

if go:
    out = run(Request(turbine=turbine, as_of=as_of), mock_toolbox(scenario))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("预测与区间")
        if out.get("forecast"):
            st.table(out["forecast"].result["forecasts"])
            if out["used_fallback"]:
                st.info("本机风速不可信，已切换后备模型")
    with col2:
        st.subheader("质量与证据")
        st.write(out["diagnosis"]["cause"]["text"])
        for f in out["diagnosis"]["findings"]:
            st.warning(f["text"])
        if not out["validation"]["ok"]:
            st.error(out["validation"]["errors"])
    st.subheader("执行轨迹")
    st.table(out["trace"])
