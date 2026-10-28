"""Train the existing GRPO adapter on tool-use demonstrations for one epoch.

Each example includes the preceding conversation and real observations.
Only the next tool request or DONE response contributes to loss.
Original model weights remain frozen; the LoRA adapter is updated.
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from baseline import MODEL, REVISION

STARTING_ADAPTER = "runs/grpo-batched-160steps/adapter"
OUTPUT_DIR = "runs/tool-sft-1epoch"

tokenizer = AutoTokenizer.from_pretrained(
    STARTING_ADAPTER,
    padding_side="right",
)

rows = json.loads(Path("data/tool_train.json").read_text())
data = Dataset.from_list([
    {"prompt": row["prompt"], "completion": row["completion"]}
    for row in rows
])

base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    revision=REVISION,
    dtype=torch.float32,
    attn_implementation="sdpa",
)

# Continue the existing adapter; the original weights remain frozen.
model = PeftModel.from_pretrained(
    base,
    STARTING_ADAPTER,
    is_trainable=True,
)
model.generation_config.eos_token_id = tokenizer.eos_token_id

config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    lr_scheduler_type="constant",
    weight_decay=0.0,
    max_length=512,
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
    model=model,
    args=config,
    train_dataset=data,
    processing_class=tokenizer,
)

assert len(trainer.train_dataset) == len(data)

# Verify both kinds of target using the trainer's actual batch builder.
done_index = next(
    i for i, row in enumerate(rows)
    if row["completion"] == "DONE" + tokenizer.eos_token
)
check_indices = [0, done_index]

batch = trainer.data_collator([
    trainer.train_dataset[i] for i in check_indices
])

for batch_index, data_index in enumerate(check_indices):
    labels = batch["labels"][batch_index]
    supervised_ids = labels[labels != -100].tolist()
    supervised_text = tokenizer.decode(supervised_ids)

    prompt_ids = tokenizer.encode(
        data[data_index]["prompt"],
        add_special_tokens=False,
    )
    assert (labels[:len(prompt_ids)] == -100).all()
    assert supervised_text == data[data_index]["completion"]

    print("Supervised text:", repr(supervised_text))
    print("Supervised tokens:", len(supervised_ids))

print("Device:", trainer.args.device)
print("Starting adapter:", STARTING_ADAPTER)
print("Training examples:", len(data))
trainer.model.print_trainable_parameters()

result = trainer.train()
trainer.save_metrics("train", result.metrics)
trainer.save_state()

adapter_path = f"{OUTPUT_DIR}/adapter"
trainer.save_model(adapter_path)
tokenizer.save_pretrained(adapter_path)

print("Saved adapter and tokenizer to:", adapter_path)