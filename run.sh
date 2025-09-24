#!/bin/bash

# 设置使用的GPU卡号，这里使用第一张卡（索引为0）
export CUDA_VISIBLE_DEVICES=0

# 检查 swanlab 是否已登录，如果没有，则提示用户登录
# 您可以预先在终端运行 `swanlab login`
if ! swanlab status | grep -q "Logged in"; then
    echo "SwanLab not logged in. Please run 'swanlab login' with your API key."
    exit 1
fi


echo "Starting training script..."
python train.py