# train.py

import os
import torch
import swanlab
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from tqdm import tqdm
import math

import config
from dataset import SC2EntityDataset
from model import Qwen2ForSC2Fusion

# --- NEW: Validation Function ---
def validate_and_log(model, val_loader, tokenizer, device, global_step):
    """
    Performs validation, logs metrics to SwanLab, and prints sample results.
    """
    print(f"\n--- Running validation at step {global_step} ---")
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    val_iterator = iter(val_loader)
    
    # --- Get the first batch for printing examples later ---
    try:
        first_val_batch = next(val_iterator)
    except StopIteration:
        print("Validation set is empty. Skipping validation.")
        model.train()
        return

    with torch.no_grad():
        # --- Process the first batch ---
        input_ids = first_val_batch['input_ids'].to(device)
        attention_mask = first_val_batch['attention_mask'].to(device)
        labels = first_val_batch['labels'].to(device)
        entity_vectors = first_val_batch['entity_vector'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            entity_vectors=entity_vectors
        )
        total_val_loss += outputs.loss.item()
        
        # --- Process remaining batches for loss calculation ---
        for batch in val_iterator:
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

    # --- Print generated examples from the first validation batch ---
    print("\n--- Validation Examples ---")
    # Print up to 2 examples from the batch
    num_examples_to_print = min(len(first_val_batch['input_ids']), 2)
    
    for i in range(num_examples_to_print):
        sample_input_ids = first_val_batch['input_ids'][i]
        sample_labels = first_val_batch['labels'][i]
        sample_entity_vector = first_val_batch['entity_vector'][i].unsqueeze(0).to(device)

        # Find where the prompt ends (where labels are not -100)
        try:
            prompt_end_index = (sample_labels != -100).nonzero(as_tuple=True)[0][0]
        except IndexError:
            prompt_end_index = len(sample_input_ids)

        prompt_ids = sample_input_ids[:prompt_end_index].unsqueeze(0).to(device)
        
        # Decode Ground Truth
        ground_truth_ids = sample_labels[prompt_end_index:]
        ground_truth_ids = ground_truth_ids[ground_truth_ids != -100]
        ground_truth_text = tokenizer.decode(ground_truth_ids, skip_special_tokens=True)
        
        # Generate Model Output
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
        
        print(f"\n--- Example {i+1} ---")
        print(f"STANDARD ANSWER: {ground_truth_text.strip()}")
        print(f"MODEL OUTPUT:    {generated_text.strip()}")
        print("-" * 20)

    print("--- End of validation ---\n")
    model.train() # Set model back to training mode


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
    # --- MODIFIED: Ensure you are using the corrected ENTITY_VECTOR_DIM from your previous debugging ---
    model = Qwen2ForSC2Fusion(model_config, config.ENTITY_VECTOR_DIM) 
    
    pretrained_dict = Qwen2ForCausalLM.from_pretrained(config.MODEL_PATH).state_dict()
    model.load_state_dict(pretrained_dict, strict=False)

    model.resize_token_embeddings(len(tokenizer))

    sc2_token_id = tokenizer.convert_tokens_to_ids(config.SC2_ENTITY_TOKEN)
    model.sc2_entity_token_id = sc2_token_id
    
    model.to(device)
    print("Model loaded successfully.")

    # --- 4. 准备数据 ---
    print(f"Loading dataset from {config.DATA_PATH}...")
    full_dataset = SC2EntityDataset(config.DATA_PATH, tokenizer)
    
    # --- MODIFIED: Split dataset into training and validation sets ---
    dataset_size = len(full_dataset)
    val_size = 15
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False) # No need to shuffle validation data
    
    print(f"Dataset split into {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # --- 5. 设置优化器 ---
    # (No changes in this section)
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
    global_step = 0 # --- NEW: Global step counter ---
    
    for epoch in range(config.EPOCHS):
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}")
        for batch in progress_bar:
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
            global_step += 1 # --- NEW: Increment step counter ---
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            swanlab.log({"training_loss": loss.item()}, step=global_step)
            
            # --- NEW: Periodic validation logic ---
            if global_step % 100 == 0:
                # --- Calculate and log gradient norm ---
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                swanlab.log({"gradient_norm": total_norm}, step=global_step)

                # --- Run validation ---
                validate_and_log(model, val_loader, tokenizer, device, global_step)

    # --- 7. 保存模型 ---
    print(f"Training finished. Saving model to {config.SAVE_PATH}...")
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)
    
    model.save_pretrained(config.SAVE_PATH)
    tokenizer.save_pretrained(config.SAVE_PATH)
    print("Model saved.")

if __name__ == "__main__":
    main()