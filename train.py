# train.py

import os
import torch
import swanlab
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from tqdm import tqdm
import math
import random

import config
from dataset import SC2EntityDataset
from model import Qwen2ForSC2Fusion

def validate_and_log(model, val_loader, tokenizer, device, global_step, num_examples_to_print=0):
    """
    执行验证，记录指标，并根据要求打印指定数量的验证样例。

    参数:
        num_examples_to_print (int): 
            - 0: 不打印任何样例。
            - > 0: 随机打印指定数量的样例。
            - -1: 打印所有样例。
    """
    print(f"\n--- Running validation at step {global_step} ---")
    model.eval()
    total_val_loss = 0
    
    if not val_loader.dataset or len(val_loader.dataset) == 0:
        print("Validation set is empty. Skipping validation.")
        model.train()
        return

    # --- 1. 计算总体验证损失 ---
    with torch.no_grad():
        for batch in val_loader:
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
            
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    swanlab.log({"validation_loss": avg_val_loss}, step=global_step)

    # --- 2. 根据要求打印样例 ---
    if num_examples_to_print != 0:
        print_all = (num_examples_to_print == -1)
        
        all_indices = list(range(len(val_loader.dataset)))
        
        if not print_all:
            num_to_sample = min(num_examples_to_print, len(all_indices))
            indices_to_print = random.sample(all_indices, num_to_sample)
            print(f"\n--- Printing {num_to_sample} random validation examples ---")
        else:
            indices_to_print = all_indices
            print("\n--- Printing ALL validation examples ---")

        with torch.no_grad():
            for i, example_index in enumerate(indices_to_print):
                sample = val_loader.dataset[example_index]
                
                sample_input_ids = sample['input_ids']
                sample_labels = sample['labels']
                sample_entity_vector = sample['entity_vector'].unsqueeze(0).to(device)

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
                
                print(f"\n--- Example {i + 1} (Dataset index: {example_index}) ---")
                print(f"STANDARD ANSWER: {ground_truth_text.strip()}")
                print(f"MODEL OUTPUT:    {generated_text.strip()}")
                print("-" * 20)
        
        print("--- End of validation examples ---")

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
    print("--- Loading Datasets ---")
    if config.VAL_DATA_PATH:
        print("Mode: Loading validation set from a separate file.")
        train_dataset = SC2EntityDataset(config.DATA_PATHS, tokenizer)
        val_dataset = SC2EntityDataset(config.VAL_DATA_PATH, tokenizer)
    else:
        print(f"Mode: Sampling {config.VAL_SAMPLES_FROM_TRAIN} examples from training data for validation.")
        full_train_dataset = SC2EntityDataset(config.DATA_PATHS, tokenizer)
        val_size = min(config.VAL_SAMPLES_FROM_TRAIN, len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size
        if train_size <= 0:
            raise ValueError("The number of validation samples to extract is >= the total dataset size.")
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    print("-------------------------")

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

    # --- 6. 加载 Checkpoint (如果存在) ---
    global_step = 0
    start_epoch = 0
    if hasattr(config, 'LOAD_CHECKPOINT_PATH') and config.LOAD_CHECKPOINT_PATH and os.path.exists(config.LOAD_CHECKPOINT_PATH):
        print(f"Loading checkpoint from {config.LOAD_CHECKPOINT_PATH}...")
        checkpoint = torch.load(config.LOAD_CHECKPOINT_PATH, map_location=device)
        
        # 使用 strict=False 允许只加载部分权重 (例如只加载MLP)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']

        print(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}, global step {global_step}.")

    # --- 7. 训练循环 ---
    print("Starting training...")
    
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", initial=global_step % len(train_loader), total=len(train_loader))
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
            
            # --- 验证、记录和保存 ---
            if global_step > 0 and global_step % config.VALIDATION_INTERVAL == 0:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                swanlab.log({"gradient_norm": total_norm}, step=global_step)

                num_to_print = 0
                if global_step % config.PRINT_ALL_VALIDATION_EXAMPLES_INTERVAL == 0:
                    num_to_print = -1
                else:
                    num_to_print = config.VALIDATION_EXAMPLES_TO_PRINT
                
                validate_and_log(
                    model, val_loader, tokenizer, device, global_step, 
                    num_examples_to_print=num_to_print
                )

            # --- 保存 Checkpoint ---
            if global_step > 0 and global_step % config.CHECKPOINT_SAVE_INTERVAL == 0:
                if not os.path.exists(config.CHECKPOINT_PATH):
                    os.makedirs(config.CHECKPOINT_PATH)
                
                checkpoint_save_path = os.path.join(config.CHECKPOINT_PATH, f"checkpoint_step_{global_step}.pt")
                print(f"\n--- Saving checkpoint at step {global_step} to {checkpoint_save_path} ---")

                save_dict = {
                    'global_step': global_step,
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                
                if config.TRAIN_MODE == "mlp_only":
                    save_dict['model_state_dict'] = {k: v for k, v in model.state_dict().items() if 'projector' in k}
                else:
                    save_dict['model_state_dict'] = model.state_dict()
                
                torch.save(save_dict, checkpoint_save_path)
                print("--- Checkpoint saved. ---")

    # --- 8. 保存最终模型 ---
    print(f"Training finished. Saving final model to {config.SAVE_PATH}...")
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)
    
    model.save_pretrained(config.SAVE_PATH)
    tokenizer.save_pretrained(config.SAVE_PATH)
    print("Model saved.")

if __name__ == "__main__":
    main()