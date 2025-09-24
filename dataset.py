# dataset.py

import json
import torch
from torch.utils.data import Dataset
import config

class SC2EntityDataset(Dataset):
    """星际争霸实体数据集"""
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            # for line in f:
            #     data.append(json.loads(line))
            data = json.load(f)
        return data

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
        # Causal LM 的 labels 就是 input_ids 的一个副本
        labels = input_ids.clone()
        
        # 将 prompt 部分的 labels 设置为 -100，这样在计算 loss 时就会被忽略
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "entity_vector": entity_vector
        }