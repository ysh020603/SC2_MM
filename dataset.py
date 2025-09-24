# dataset.py

import json
import torch
from torch.utils.data import Dataset
import config

class SC2EntityDataset(Dataset):
    """星际争霸实体数据集"""
    def __init__(self, data_paths, tokenizer): # <-- 参数名从 data_path 改为 data_paths
        self.tokenizer = tokenizer
        # --- MODIFIED: 调用 _load_data 加载多个文件 ---
        self.data = self._load_data(data_paths)
        
    def _load_data(self, data_paths): # <-- 参数名从 data_path 改为 data_paths
        """从文件路径列表中加载并合并数据。"""
        # --- MODIFIED: 检查输入是单个文件还是列表 ---
        # 这样做可以保持代码的向后兼容性，如果传入的是单个字符串，也能正常工作
        if isinstance(data_paths, str):
            paths = [data_paths]
        else:
            paths = data_paths

        all_data = []
        # --- MODIFIED: 循环读取所有文件 ---
        print(f"Loading data from: {paths}")
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                # 假设每个文件都是一个包含多个对象的JSON数组
                data = json.load(f)
                all_data.extend(data)
        print(f"Total records loaded: {len(all_data)}")
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        response = item['response']
        entity_vector = torch.tensor(item['code'], dtype=torch.float32)

        # 将 prompt 和 response 拼接起来进行训练
        full_text = prompt + response + self.tokenizer.eos_token
        
        # 对 prompt 单独编码，以计算 loss 时需要忽略的部分
        prompt_tokenized = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_len = prompt_tokenized.input_ids.shape[1]

        # 对全文编码
        tokenized_output = self.tokenizer(
            full_text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokenized_output.input_ids.squeeze(0)
        attention_mask = tokenized_output.attention_mask.squeeze(0)

        # 创建 labels，用于计算 loss
        labels = input_ids.clone()
        
        # 将 prompt 部分的 labels 设置为 -100
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "entity_vector": entity_vector
        }