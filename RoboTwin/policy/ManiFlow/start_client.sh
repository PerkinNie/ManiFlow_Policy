#!/bin/bash

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "ManiFlow推理客户端启动脚本"
echo "=========================================="

python policy/ManiFlow/inference_framework/client.py \
    --host "localhost" \
    --port 5000 \
    --max-steps 100 \
    --task-name "dual_arm_pick_box"
