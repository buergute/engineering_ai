#!/usr/bin/env bash
# 合成数据仅用于验证流程，不用于正式报告。
set -euo pipefail
cd "$(dirname "$0")/.."
run_step() {
  local label="$1" output="$2" start=$SECONDS
  shift 2
  echo "== ${label}"
  "$@"
  echo "耗时：$((SECONDS-start)) 秒；产物：${output}"
}
run_step '1/7 测试' '无' python -m pytest -q
run_step '2/7 生成合成数据' 'data/synth/' python -m datagen.synth
run_step '3/7 训练与校准' 'artifacts/' python -m models.train
run_step '4/7 欠发和复核阈值' 'artifacts/thresholds.json' python -m eval.thresholds
run_step '5/7 预测评测（表 1）' 'outputs/numeric_results.md' python -m eval.numeric
run_step '6/7 检测与复核（表 2）' 'outputs/review_results.md' python -m eval.review_eval --n 40
run_step '7/7 智能体任务（表 3 底稿）' 'outputs/task_results.md' python -m agent_eval.run_tasks
