# dataset.py

import json
import torch
from torch.utils.data import Dataset
import random  # <-- 新增导入
import config  # <-- 新增导入

class SC2EntityDataset(Dataset):
    """星际争霸实体数据集"""
    def __init__(self, data_paths, tokenizer): # <-- 参数名从 data_path 改为 data_paths
        self.tokenizer = tokenizer
        # --- MODIFIED: 调用 _load_data 加载多个文件 ---
        self.data = self._load_data(data_paths)
        
    def _load_data(self, data_paths):
        """
        根据 config.py 中的设置加载数据。
        如果 config.USE_PROPORTIONAL_SAMPLING 为 True，则按比例从每个文件中采样。
        否则，加载所有文件中的全部数据。
        """
        if isinstance(data_paths, str):
            paths = [data_paths]
        else:
            paths = data_paths

        # --- 新增：按比例采样逻辑 ---
        if hasattr(config, 'USE_PROPORTIONAL_SAMPLING') and config.USE_PROPORTIONAL_SAMPLING:
            print("--- Proportional sampling is ENABLED ---")
            
            # 校验配置的正确性
            if len(paths) != len(config.DATA_PROPORTIONS):
                raise ValueError("Error: The number of data paths in config.DATA_PATHS must match the number of proportions in config.DATA_PROPORTIONS.")
            if not torch.isclose(torch.tensor(config.DATA_PROPORTIONS).sum(), torch.tensor(1.0)):
                raise ValueError(f"Error: The sum of proportions in config.DATA_PROPORTIONS must be 1.0, but it is {sum(config.DATA_PROPORTIONS)}.")

            all_data = []
            total_samples = config.TOTAL_TRAIN_SAMPLES
            
            print(f"Target total samples for training: {total_samples}")

            # 遍历每个文件路径和其对应的比例
            for path, proportion in zip(paths, config.DATA_PROPORTIONS):
                num_samples_to_take = int(total_samples * proportion)
                
                if num_samples_to_take == 0:
                    print(f"Skipping '{path}' as the calculated number of samples is 0.")
                    continue

                print(f"Loading from '{path}' to sample {num_samples_to_take} records...")
                with open(path, 'r', encoding='utf-8') as f:
                    # 首先加载文件中的全部数据
                    full_data_from_file = json.load(f)
                
                # 如果请求的样本数大于文件中的总数，发出警告并使用文件中的所有数据
                if num_samples_to_take > len(full_data_from_file):
                    print(f"  Warning: Requested {num_samples_to_take} samples, but file only contains {len(full_data_from_file)}. Using all available data from this file.")
                    num_samples_to_take = len(full_data_from_file)

                # 随机打乱列表，然后抽取所需数量的样本
                random.shuffle(full_data_from_file)
                sampled_data = full_data_from_file[:num_samples_to_take]
                all_data.extend(sampled_data)

            print("-" * 20)
            print(f"Total records loaded after sampling: {len(all_data)}")
            print("--- Proportional sampling finished ---")
            return all_data

        # --- 原始逻辑：加载所有数据 ---
        else:
            print("--- Proportional sampling is DISABLED. Loading all data from all files. ---")
            all_data = []
            print(f"Loading data from: {paths}")
            for path in paths:
                with open(path, 'r', encoding='utf-8') as f:
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