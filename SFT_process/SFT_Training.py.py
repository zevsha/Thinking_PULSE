
import os
import torch
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback
import matplotlib.pyplot as plt
import numpy as np
import json
import math


os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

df = pd.read_json("Dataset path", lines=True)
ds = Dataset.from_pandas(df)


repo_id = "microsoft/Phi-4-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    device_map="auto",
    quantization_config=bnb_config,
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["o_proj", "qkv_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)


tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(example):
    full = example["input"] + " " + example["output"]
    tokenized = tokenizer(full, truncation=True, padding="longest", max_length=512)
    inp_len = len(tokenizer(example["input"], add_special_tokens=False)["input_ids"])
    labels = [-100] * inp_len + tokenized["input_ids"][inp_len:]
    tokenized["labels"] = labels[:512]
    return tokenized

ds = ds.map(tokenize_fn, remove_columns=["input", "output"], batched=False)


sft_config = SFTConfig(
    output_dir="output model directory",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    num_train_epochs=7,
    learning_rate=2e-4,
    logging_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=1,
    fp16=True,
    optim="paged_adamw_32bit",
)




class MetricsLoggerCallback(TrainerCallback):
    def __init__(self):
        self.metrics = {"epochs": [], "losses": [], "mean_token_acc": []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and "epoch" in logs:
            self.metrics["epochs"].append(logs["epoch"])
            self.metrics["losses"].append(logs["loss"])
            # Use math.exp for float
            self.metrics["mean_token_acc"].append(float(math.exp(-logs["loss"])))
        return control


metrics_logger = MetricsLoggerCallback()


trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    peft_config=peft_config,
    args=sft_config,
    callbacks=[metrics_logger],
)


trainer.train(resume_from_checkpoint=False)


model.save_pretrained(sft_config.output_dir)
tokenizer.save_pretrained(sft_config.output_dir)


metrics_file = os.path.join(sft_config.output_dir, "metrics_log.json")
with open(metrics_file, "w") as f:
    json.dump(metrics_logger.metrics, f, indent=4)


unique_epochs = sorted(set(metrics_logger.metrics["epochs"]))
avg_losses = [
    np.mean([metrics_logger.metrics["losses"][i] for i, e in enumerate(metrics_logger.metrics["epochs"]) if e == ue])
    for ue in unique_epochs
]

plt.figure(figsize=(8,5))
plt.plot(unique_epochs, avg_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.savefig(os.path.join(sft_config.output_dir, "loss_vs_epoch.png"))
plt.close()


avg_acc = [
    np.mean([metrics_logger.metrics["mean_token_acc"][i] for i, e in enumerate(metrics_logger.metrics["epochs"]) if e == ue])
    for ue in unique_epochs
]

plt.figure(figsize=(8,5))
plt.plot(unique_epochs, avg_acc, marker='o', color='green')
plt.xlabel("Epoch")
plt.ylabel("Mean Token Accuracy")
plt.title("Mean Token Accuracy vs Epoch")
plt.grid(True)
plt.savefig(os.path.join(sft_config.output_dir, "mean_token_acc_vs_epoch.png"))
plt.close()
