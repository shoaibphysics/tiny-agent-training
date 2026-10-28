"""Fine-tune a fresh LoRA adapter for three epochs.

Learn from successful number-game demonstrations.
Only answer tokens contribute to loss; original model weights stay frozen.
Save the adapter, tokenizer and training measurements.
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from baseline import MODEL, REVISION, make_prompt

OUTPUT_DIR = "runs/sft-3epochs"

tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)

puzzles = json.loads(Path("data/train.json").read_text())
data = Dataset.from_list([
    {
        "prompt": make_prompt(tokenizer, row["start"], row["target"]),
        "completion": row["completion"] + tokenizer.eos_token,
    }
    for row in puzzles
])

adapter = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
    revision=REVISION,
)

config = SFTConfig(
    output_dir=OUTPUT_DIR,
    model_init_kwargs={
        "revision": REVISION,
        "dtype": torch.float32,
        "attn_implementation": "sdpa",
    },
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    lr_scheduler_type="constant",
    max_length=256,
    completion_only_loss=True,
    packing=False,
    padding_free=False,
    bf16=False,
    fp16=False,
    gradient_checkpointing=False,
    optim="adamw_torch",
    dataloader_pin_memory=False,
    logging_steps=20,
    save_strategy="no",
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=MODEL,
    args=config,
    train_dataset=data,
    processing_class=tokenizer,
    peft_config=adapter,
)

# Verify the trainer's supervised targets before training.
batch = trainer.data_collator([trainer.train_dataset[0]])
labels = batch["labels"][0]
supervised_ids = labels[labels != -100].tolist()
supervised_text = tokenizer.decode(supervised_ids)

assert supervised_text == data[0]["completion"]
print("Supervised text:", repr(supervised_text))
print("Supervised tokens:", len(supervised_ids))
print("Device:", trainer.args.device)
trainer.model.print_trainable_parameters()

# Train and preserve measurements for analysis and the README.
result = trainer.train()
trainer.save_metrics("train", result.metrics)
trainer.save_state()

adapter_path = f"{OUTPUT_DIR}/adapter"
trainer.save_model(adapter_path)
tokenizer.save_pretrained(adapter_path)
print("Saved adapter and tokenizer to:", adapter_path)