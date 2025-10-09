"""LoRA fine-tuning.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

TRAIN_FILE_PATH = 'data/train_dataset.json'
VAL_FILE_PATH = 'data/val_datast.json'

BASE_MODEL = 't-tech/T-pro-it-1.0'

OUTPUT_PATH = 'lora_model'

train_dataset = Dataset.from_json(TRAIN_FILE_PATH)
val_dataset = Dataset.from_json(VAL_FILE_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
)

model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

if hasattr(model, 'generation_config') and model.generation_config is not None:
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

model.gradient_checkpointing_enable()

# model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.15,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)

args = SFTConfig(
    output_dir=OUTPUT_PATH,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    eval_accumulation_steps=16,
    learning_rate=1e-5,
    lr_scheduler_type='cosine',
    weight_decay=0.1,
    warmup_ratio=0.15,
    max_length=2048,
    eval_strategy="steps",  # "epoch"
    save_strategy="steps",  # "epoch"
    eval_steps=300,
    save_steps=300,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=16,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
