#!/usr/bin/env bash
#
# 依次运行 config.json 中除 base 外的 6 组控制变量实验。
# 每组 3 个 epoch，在 Apple Silicon 上约 3~10 分钟，全部跑完约 30~60 分钟。
#
# 用法：
#   bash scripts/run_experiments.sh
#   PYTHON=/opt/anaconda3/envs/learn-pytorch/bin/python bash scripts/run_experiments.sh

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"

EXPERIMENTS=(
    exp_lr_1e-5
    exp_lr_1e-4
    exp_batch_4
    exp_batch_64
    exp_dropout_0.03
    exp_dropout_0.3
)

echo "项目根目录：$PROJECT_ROOT"
echo "Python    ：$("$PYTHON" -c 'import sys; print(sys.executable)')"
echo

failed=()
for exp in "${EXPERIMENTS[@]}"; do
    echo "=============================================================="
    echo ">>> $exp"
    echo "=============================================================="
    if ! "$PYTHON" src/train.py --exp "$exp"; then
        echo "!!! $exp 运行失败"
        failed+=("$exp")
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
