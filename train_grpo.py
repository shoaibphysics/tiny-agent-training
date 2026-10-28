"""Continue the SFT adapter using batched GRPO with rollout reuse.

Collect 4 puzzles x 8 plans = 32 plans per rollout batch.
Train in minibatches of 8, making two passes through that batch.
After 8 optimizer updates, collect fresh plans from the updated model.
Run 20 collection cycles: 160 optimizer updates in total.
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from baseline import MODEL, REVISION, MAX_NEW_TOKENS, make_prompt
from game import score

SFT_ADAPTER = "runs/sft-3epochs/adapter"
OUTPUT_DIR = "runs/grpo-batched-160steps"

tokenizer = AutoTokenizer.from_pretrained(
    SFT_ADAPTER,
    padding_side="left",
)

base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    revision=REVISION,
    dtype=torch.float32,
    attn_implementation="sdpa",
)

# Load SFT once. Continue updating this adapter throughout the run.
model = PeftModel.from_pretrained(
    base,
    SFT_ADAPTER,
    is_trainable=True,
)

model.generation_config.eos_token_id = tokenizer.eos_token_id

puzzles = json.loads(Path("data/train.json").read_text())

data = Dataset.from_list([
    {
        "prompt": make_prompt(tokenizer, row["start"], row["target"]),
        "start": row["start"],
        "target": row["target"],
    }
    for row in puzzles
])


def game_reward(completions, start, target, trainer_state, **kwargs):
    rewards = [
        score(answer, s, t)
        for answer, s, t in zip(completions, start, target, strict=True)
    ]

    # Called when collecting fresh plans, not on every reuse update.
    print(
        f"\nCollected {len(rewards)} plans at optimizer step "
        f"{trainer_state.global_step} | "
        f"mean reward={sum(rewards) / len(rewards):.3f}"
    )
    return rewards


config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    max_steps=160,

    # Collection: 4 puzzles, with 8 plans for each puzzle.
    generation_batch_size=32,
    num_generations=8,

    # Optimization: 2 microbatches of 4 per optimizer update.
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,

    # Two passes through each collected batch.
    num_iterations=2,
    shuffle_dataset=True,

    max_completion_length=MAX_NEW_TOKENS,
    temperature=0.7,
    top_p=1.0,
    top_k=0,

    learning_rate=1e-5,
    lr_scheduler_type="constant",
    weight_decay=0.0,
    loss_type="grpo",
    scale_rewards="group",
    epsilon=0.2,
    beta=0.0,

    remove_unused_columns=False,
    use_vllm=False,
    bf16=False,
    fp16=False,
    gradient_checkpointing=False,
    optim="adamw_torch",
    dataloader_pin_memory=False,

    # Report after each complete collection-and-training cycle.
    logging_steps=8,
    save_strategy="no",
    report_to="none",
    seed=42,
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=data,
    reward_funcs=game_reward,
    processing_class=tokenizer,
)

print("Device:", trainer.args.device)
print("Starting adapter:", SFT_ADAPTER)
trainer.model.print_trainable_parameters()

result = trainer.train()
trainer.save_metrics("train", result.metrics)
trainer.save_state()

adapter_path = f"{OUTPUT_DIR}/adapter"
trainer.save_model(adapter_path)
tokenizer.save_pretrained(adapter_path)

print("Saved adapter and tokenizer to:", adapter_path)