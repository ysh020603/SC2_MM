# inference.py (修复后版本)

import torch
import numpy as np
from transformers import AutoTokenizer

# 导入项目特定的模块
import config
from model import Qwen2ForSC2Fusion # [关键] 必须导入自定义的模型类

def generate_response(prompt: str, entity_vector: list, model: Qwen2ForSC2Fusion, tokenizer: AutoTokenizer) -> str:
    """
    使用您的 SC2Fusion 模型生成回复。

    参数:
        prompt (str): 包含特殊 token 的输入提示。
        entity_vector (list): 实体属性的向量表示。
        model (Qwen2ForSC2Fusion): 已加载的、训练好的模型实例。
        tokenizer (AutoTokenizer): 已加载的分词器实例。

    返回:
        str: 模型生成的回复文本。
    """
    device = model.device

    # --- 1. 准备输入 ---
    # 对输入的文本提示进行分词
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # 将实体向量转换为 PyTorch 张量，并增加一个批次维度 (batch dimension)
    # [1, ENTITY_VECTOR_DIM]
    entity_vector_tensor = torch.tensor(entity_vector, dtype=torch.float32).unsqueeze(0).to(device)

    # --- 2. 生成回复 ---
    # 在不计算梯度的模式下执行，以节省计算资源和加速
    print("正在生成回复...")
    with torch.no_grad():
        # 调用 model.generate 函数
        # Transformers 的 generate 函数非常灵活，它会自动将我们传入的未知参数 (如此处的 entity_vectors)
        # 传递给模型底层的 forward 方法。这正是我们自定义模型所需要的。
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_vectors=entity_vector_tensor,  # [关键] 传递自定义的实体向量
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # --- 3. 解码并返回结果 ---
    # output_ids 包含输入的 prompt 部分，我们需要将其切片去掉，只解码新生成的部分
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

def main():
    """
    主执行函数
    """
    # --- 1. 加载训练好的模型和分词器 ---
    # [FIX] 修改模型加载路径以匹配报错信息中的路径
    save_path = '/data4/SC2/SC2_units_token_compress/model/Qwen_1_5B_MLP_modify_4_epochs'
    print(f"正在从 '{save_path}' 加载已训练的模型...")

    # 加载分词器
    # trust_remote_code=True 是因为 Qwen 模型需要
    # local_files_only=True 确保只从本地加载，避免不必要的网络请求
    # use_fast=False 是本次修复的关键，它会强制使用 Python 版的 Tokenizer，绕过损坏的 tokenizer.json 文件问题
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            save_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False  # <--- [核心修复] 添加此行
        )
    except Exception as e:
        print(f"加载分词器时出错: {e}")
        print("如果问题仍然存在，请尝试删除模型保存目录下的 'tokenizer.json' 文件，然后重新运行。")
        return

    # 加载自定义模型
    # 同样，我们使用 from_pretrained 方法，但作用于我们自定义的 Qwen2ForSC2Fusion 类
    # 这样 Transformers 库会自动加载权重并实例化我们的类
    model = Qwen2ForSC2Fusion.from_pretrained(
        save_path,
        entity_vector_dim=config.ENTITY_VECTOR_DIM, # [关键] 需要提供我们自定义的参数
        trust_remote_code=True,
        local_files_only=True
    )

    # [重要] 确保模型知道特殊 token 的 ID
    # 在 forward 方法中，模型需要根据这个 ID 找到插入实体向量的位置
    sc2_token_id = tokenizer.convert_tokens_to_ids(config.SC2_ENTITY_TOKEN)
    model.sc2_entity_token_id = sc2_token_id

    # 将模型移动到配置的设备 (CPU 或 GPU) 并设置为评估模式
    device = torch.device(config.DEVICE)
    model.to(device)
    model.eval() # 评估模式会关闭 Dropout 等层
    print("模型加载成功！")

    # --- 2. 准备测试数据 ---
    # 这个 prompt 必须包含在训练时使用的特殊 token <sc2_entity>
    # test_prompt = (
    #     f"{config.SC2_ENTITY_TOKEN}\n"
    #     "这是一个星际争霸场景中的单位。该单位的属性包括：\n"
    #     "1. 是否为友方单位\n"
    #     "2. 单位名称\n"
    #     "3. 单位的生命值\n"
    #     "4. 单位的位置\n\n"
    #     "请用结构化的方式输出该单位的属性。\n"
    #     "例如: [True, # 友方单位，如果不是则为 False\n"
    #     "\"xxxxx\", # 单位名称\n"
    #     "100, # 单位的生命值\n"
    #     "{\"x\": 10, \"y\": 20}] # 单位的位置\n"
    # )

    test

    # 创建一个符合维度的伪实体向量用于测试
    # 在真实应用中，这个向量应该由一个编码器根据单位的真实属性生成
    print(f"实体向量维度: {config.ENTITY_VECTOR_DIM}")
    test_entity_vector = np.random.rand(config.ENTITY_VECTOR_DIM).tolist()

    # --- 3. 执行推理 ---
    response = generate_response(test_prompt, test_entity_vector, model, tokenizer)

    # --- 4. 打印结果 ---
    print("\n" + "="*50)
    print("输入 Prompt:")
    print(test_prompt)
    print("\n" + "="*50)
    print("模型生成结果:")
    print(response)
    print("="*50)


if __name__ == "__main__":
    main()