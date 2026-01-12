#!/bin/bash

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "ManiFlow推理服务端启动脚本"
echo "=========================================="

python policy/ManiFlow/inference_framework/server.py \
    --host "0.0.0.0" \
    --port 5000 \
    --device "cuda:0"
