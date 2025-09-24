# config.py

import torch

# --- 路径配置 ---
# 本地预训练模型路径
MODEL_PATH = "/data2/shy/model/Qwen/Qwen2___5-1___5B-Instruct/"
# MODEL_PATH = "/data2/shy/model/Qwen/Qwen2___5-7B-Instruct/"

# 训练数据文件路径 (请确保您创建了这个文件)
DATA_PATH = "/data4/SC2/train_data_prompt_modify.json"

# 训练后模型权重保存路径
SAVE_PATH = "/data4/SC2/SC2_units_token_compress/model/Qwen_1_5B_MLP_modify_4_epochs"

# --- 模型配置 ---
# 新添加的特殊 token
SC2_ENTITY_TOKEN = "<sc2_entity>"

# 实体向量的维度 (根据您的数据)
ENTITY_VECTOR_DIM = 83

# --- 训练配置 ---
# 训练设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 训练模式:
# "mlp_only" - 只训练投影层MLP
# "full" - 训练MLP和整个LLM
TRAIN_MODE = "mlp_only"

# 学习率
LEARNING_RATE = 2e-5

# 批处理大小
BATCH_SIZE = 18

# 训练轮数
EPOCHS = 3

# --- SwanLab 配置 ---
# SwanLab 项目名称
SWANLAB_PROJECT_NAME = "Qwen2-SC2-Fusion"

# SwanLab 实验名称
SWANLAB_EXPERIMENT_NAME = "train-Qwen_1.5B_MLP_modify_4_epochs"