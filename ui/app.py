"""界面（负责人：C）。运行：streamlit run ui/app.py

四块：任务与起报时间 / 预测与区间 / 质量与证据 / 执行轨迹。
数据源可选“合成数据上的真实工具”（需先跑 scripts/run_all.sh）或“假工具”。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from agent.mock_tools import SCENARIOS, mock_toolbox  # noqa: E402
from agent.report import VERDICT_CN  # noqa: E402
from agent.state_machine import Request, run  # noqa: E402
from agent.toolbox import real_toolbox  # noqa: E402

st.set_page_config(page_title="风电预测与复核智能体", layout="wide")
st.title("风电功率预测与异常复核智能体")
ready = (ROOT / "artifacts" / "models.pkl").exists() and (ROOT / "data" / "synth" / "scada.parquet").exists()

with st.sidebar:
    st.header("任务与起报时间")
    text = st.text_area("请求", "预测 T03 未来一小时的功率，并解释它为什么比相邻机组低")
    as_of = st.text_input("起报时刻", "2023-03-04T14:00")
    source = st.radio("数据源", ["合成数据 + 真实工具", "假工具"], index=0 if ready else 1)
    scenario = st.selectbox("假工具场景", SCENARIOS, disabled=source != "假工具")
    go = st.button("运行", type="primary")


@st.cache_resource
def _real():
    return real_toolbox()


if go:
    if source != "假工具" and not ready:
        st.error("还没有合成数据或模型，请先运行 bash scripts/run_all.sh")
        st.stop()
    tools = mock_toolbox(scenario) if source == "假工具" else _real()
    out = run(Request(text, as_of or None), tools)
    st.subheader(f"结论：{VERDICT_CN[out['verdict']]}")
    if out["message"]:
        st.info(out["message"])
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 预测与区间（MW）")
            if out["forecast"]:
                st.table(pd.DataFrame(out["forecast"].result["forecasts"]))
                if out["used_fallback"]:
                    st.info("本机风速不可信，已切换后备模型")
        with col2:
            st.markdown("#### 质量与证据")
            st.write(out["diagnosis"]["text"])
            for f in out["diagnosis"]["findings"]:
                st.warning(f["text"])
            if not out["validation"]["ok"]:
                st.error(out["validation"]["errors"])
    st.markdown("#### 执行轨迹")
    st.table(pd.DataFrame(out["trace"]).astype(str))
