#!/usr/bin/env bash
#
# 依次运行 src/experiments/ 下的 6 组控制变量实验。
# 每组 3 个 epoch，在 Apple Silicon 上约 6~12 分钟，全部跑完约 1 小时。
#
# 用法：
#   bash scripts/run_experiments.sh
#   PYTHON=/opt/anaconda3/envs/learn-pytorch/bin/python bash scripts/run_experiments.sh

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"

echo "项目根目录：$PROJECT_ROOT"
echo "Python    ：$("$PYTHON" -c 'import sys; print(sys.executable)')"
echo

failed=()
for script in src/experiments/exp_*.py; do
    echo "=============================================================="
    echo ">>> $script"
    echo "=============================================================="
    if ! "$PYTHON" "$script"; then
        echo "!!! $script 运行失败"
        failed+=("$script")
    fi
    echo
done

echo "=============================================================="
if [ ${#failed[@]} -eq 0 ]; then
    echo "全部实验完成。运行 tensorboard --logdir runs 查看对比曲线。"
else
    echo "以下实验失败："
    printf '  - %s\n' "${failed[@]}"
    exit 1
fi
