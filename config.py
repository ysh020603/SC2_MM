# config.py

import torch

# 训练后模型权重保存路径
SAVE_PATH = "Qwen_3B_MLP_diff_prompt"

# --- Checkpoint 配置 ---
# Checkpoint 保存路径
CHECKPOINT_PATH = f"{SAVE_PATH}/checkpoints"
# 是否在训练时加载 Checkpoint (提供 Checkpoint 的完整路径)
# 例如: "checkpoints/checkpoint_step_1000.pt"
LOAD_CHECKPOINT_PATH = None
# 每隔多少步保存一次 Checkpoint
CHECKPOINT_SAVE_INTERVAL = 3000


MODEL_PATH = "/9950backfile/zhangwt/model_llm/Qwen2___5-3B-Instruct"

# 训练数据文件路径列表
DATA_PATHS = [
    # "/9950backfile/zhangwt/SC2_data/output_0_to_10000_train.json",
    # "/9950backfile/zhangwt/SC2_data/output_10000_to_20000_train.json",
    # "/9950backfile/zhangwt/SC2_data/output_20000_to_30000_train.json",
    # "/9950backfile/zhangwt/SC2_data/output_30000_to_40000_train.json",
    # "/9950backfile/zhangwt/SC2_data/output_40000_to_50000_train.json",
    # "/9950backfile/zhangwt/SC2_data/output_50000_to_60000_train.json",
    # "/9950backfile/zhangwt/SC2_data/train_data_prompt_modify.json",
    # "/9950backfile/zhangwt/SC2_data/train_data_prompt_only_flag_modify.json",
    # "/9950backfile/zhangwt/SC2_data/train_data_description_1_train.json",
    # "/9950backfile/zhangwt/SC2_data/train_data_description_3_train.json",
    # "/9950backfile/zhangwt/SC2_data/train_data_description_2_train.json",
    "/9950backfile/zhangwt/SC2_data/train_data_prompt_diff_100.json"
]

# 验证数据文件路径 (可选)
# 1. 如果提供一个文件路径, 则会从该文件加载验证集。
# 2. 如果设置为 None 或者空字符串 (""), 则会自动从训练数据中抽取 VAL_SAMPLES_FROM_TRAIN 条数据作为验证集。
# VAL_DATA_PATH = "/9950backfile/shy/SC2_code/val_data_merge.json"
VAL_DATA_PATH = None

# 当 VAL_DATA_PATH 为 None 时，从训练集中抽取的样本数量
VAL_SAMPLES_FROM_TRAIN = 100

# --- 验证与日志配置 ---
# 每隔多少步进行一次验证
VALIDATION_INTERVAL = 100
# 在验证时，随机打印多少个样例
VALIDATION_EXAMPLES_TO_PRINT = 5
# 每隔多少步，打印所有的验证样例
PRINT_ALL_VALIDATION_EXAMPLES_INTERVAL = 1000



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
BATCH_SIZE = 10

# 训练轮数
EPOCHS = 3

# --- SwanLab 配置 ---
# SwanLab 项目名称
SWANLAB_PROJECT_NAME = "Qwen2-SC2-Fusion-105"

# SwanLab 实验名称
SWANLAB_EXPERIMENT_NAME = f"train_{SAVE_PATH}"