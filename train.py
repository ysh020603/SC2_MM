# train.py

import os
import torch
import swanlab
from torch.utils.data import DataLoader, random_split # 确保导入 random_split
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from tqdm import tqdm
import math

import config
from dataset import SC2EntityDataset
from model import Qwen2ForSC2Fusion

# ... (validate_and_log 函数保持不变) ...
def validate_and_log(model, val_loader, tokenizer, device, global_step, print_all_examples=False):
    """
    Performs validation, logs metrics, and optionally prints all validation examples.
    """
    print(f"\n--- Running validation at step {global_step} ---")
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    
    if not val_loader.dataset:
        print("Validation set is empty. Skipping validation.")
        model.train()
        return

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            entity_vectors = batch['entity_vector'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                entity_vectors=entity_vectors
            )
            total_val_loss += outputs.loss.item()
            
            if print_all_examples:
                if batch_idx == 0:
                    print("\n--- Printing ALL Validation Examples ---")
                
                num_examples_in_batch = len(batch['input_ids'])
                for i in range(num_examples_in_batch):
                    sample_input_ids = batch['input_ids'][i]
                    sample_labels = batch['labels'][i]
                    sample_entity_vector = batch['entity_vector'][i].unsqueeze(0).to(device)

                    try:
                        prompt_end_index = (sample_labels != -100).nonzero(as_tuple=True)[0][0]
                    except IndexError:
                        prompt_end_index = len(sample_input_ids)

                    prompt_ids = sample_input_ids[:prompt_end_index].unsqueeze(0).to(device)
                    
                    ground_truth_ids = sample_labels[prompt_end_index:]
                    ground_truth_ids = ground_truth_ids[ground_truth_ids != -100]
                    ground_truth_text = tokenizer.decode(ground_truth_ids, skip_special_tokens=True)
                    
                    generated_ids = model.generate(
                        input_ids=prompt_ids,
                        entity_vectors=sample_entity_vector,
                        max_new_tokens=256,
                        num_beams=2,
                        early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    generated_text = tokenizer.decode(generated_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True)
                    
                    example_index = batch_idx * val_loader.batch_size + i + 1
                    print(f"\n--- Example {example_index} ---")
                    print(f"STANDARD ANSWER: {ground_truth_text.strip()}")
                    print(f"MODEL OUTPUT:    {generated_text.strip()}")
                    print("-" * 20)
            
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    swanlab.log({"validation_loss": avg_val_loss}, step=global_step)
    
    if print_all_examples:
        print("--- End of all validation examples ---")

    print("--- End of validation ---\n")
    model.train()

def main():
    # --- 1. 初始化 ---
    print("Initializing...")
    swanlab.init(
        project=config.SWANLAB_PROJECT_NAME,
        experiment_name=config.SWANLAB_EXPERIMENT_NAME,
        config=vars(config)
    )
    
    device = torch.device(config.DEVICE)

    # --- 2. 加载 Tokenizer ---
    print(f"Loading tokenizer from {config.MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, trust_remote_code=True)
    
    special_tokens_dict = {'additional_special_tokens': [config.SC2_ENTITY_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. 加载模型 ---
    print(f"Loading model from {config.MODEL_PATH}...")
    model_config = Qwen2Config.from_pretrained(config.MODEL_PATH)
    model = Qwen2ForSC2Fusion(model_config, config.ENTITY_VECTOR_DIM) 
    
    pretrained_dict = Qwen2ForCausalLM.from_pretrained(config.MODEL_PATH).state_dict()
    model.load_state_dict(pretrained_dict, strict=False)

    model.resize_token_embeddings(len(tokenizer))

    sc2_token_id = tokenizer.convert_tokens_to_ids(config.SC2_ENTITY_TOKEN)
    model.sc2_entity_token_id = sc2_token_id
    
    model.to(device)
    print("Model loaded successfully.")

    # --- 4. 准备数据 ---
    # --- MODIFIED: 根据配置决定如何创建验证集 ---
    print("--- Loading Datasets ---")
    
    if config.VAL_DATA_PATH:
        # 模式一：从独立文件加载训练集和验证集
        print("Mode: Loading validation set from a separate file.")
        train_dataset = SC2EntityDataset(config.DATA_PATHS, tokenizer)
        val_dataset = SC2EntityDataset(config.VAL_DATA_PATH, tokenizer)
    else:
        # 模式二：从训练集中抽样作为验证集
        print(f"Mode: Sampling {config.VAL_SAMPLES_FROM_TRAIN} examples from training data for validation.")
        full_train_dataset = SC2EntityDataset(config.DATA_PATHS, tokenizer)
        
        # 确保抽样数量不超过数据集总数
        val_size = min(config.VAL_SAMPLES_FROM_TRAIN, len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size
        
        if train_size <= 0:
            raise ValueError("The number of validation samples to extract is greater than or equal to the total dataset size. Please reduce VAL_SAMPLES_FROM_TRAIN.")
            
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    print("-------------------------")
    # --- END OF MODIFICATION ---

    # --- 5. 设置优化器 ---
    optimizer = None
    if config.TRAIN_MODE == "mlp_only":
        print("Training mode: MLP projector only.")
        for name, param in model.named_parameters():
            if "projector" not in name:
                param.requires_grad = False
        optimizer = torch.optim.AdamW(model.projector.parameters(), lr=config.LEARNING_RATE)
    elif config.TRAIN_MODE == "full":
        print("Training mode: Full model fine-tuning.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    else:
        raise ValueError("Invalid TRAIN_MODE in config.py. Choose 'mlp_only' or 'full'.")

    # --- 6. 训练循环 ---
    print("Starting training...")
    global_step = 0
    
    for epoch in range(config.EPOCHS):
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}")
        for batch in progress_bar:
            if not (batch['input_ids'] == model.sc2_entity_token_id).any():
                print(f"\nWarning: Skipping a batch at step {global_step} because it contains no SC2_ENTITY_TOKEN.")
                continue

            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            entity_vectors = batch['entity_vector'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                entity_vectors=entity_vectors
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            global_step += 1
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            swanlab.log({"training_loss": loss.item()}, step=global_step)
            
            if global_step > 0 and global_step % 100 == 0:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                swanlab.log({"gradient_norm": total_norm}, step=global_step)

                should_print_all = (global_step % 1000 == 0)
                
                validate_and_log(
                    model, 
                    val_loader, 
                    tokenizer, 
                    device, 
                    global_step, 
                    print_all_examples=should_print_all
                )

    # --- 7. 保存模型 ---
    print(f"Training finished. Saving model to {config.SAVE_PATH}...")
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)
    
    model.save_pretrained(config.SAVE_PATH)
    tokenizer.save_pretrained(config.SAVE_PATH)
    print("Model saved.")

if __name__ == "__main__":
    main()