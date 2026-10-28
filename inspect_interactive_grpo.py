"""Inspect sampled interactive episodes before GRPO training.

Sample eight complete episodes per training puzzle.
Check whether rewards differ within each group.
This script performs inference only; model weights do not change.
"""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl.chat_template_utils import add_response_schema

from baseline import MODEL, REVISION
from evaluate_interaction import (
    MAX_NEW_TOKENS,
    MAX_TURNS,
    play_episode,
)

ADAPTER_PATH = "runs/tool-sft-1epoch/adapter"
NUM_PUZZLES = 12
GROUP_SIZE = 8
TEMPERATURE = 0.7
SEED = 42

# Seed once so subsequent episodes can draw different samples.
set_seed(SEED)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    padding_side="left",
)
tokenizer = add_response_schema(tokenizer)

base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    revision=REVISION,
    dtype=torch.float32,
).to(device)

model = PeftModel.from_pretrained(
    base,
    ADAPTER_PATH,
    is_trainable=False,
)
model.eval()

# Use training puzzles only. Their reference solutions are not used.
puzzles = json.loads(Path("data/train.json").read_text())[:NUM_PUZZLES]

groups = []
mixed_groups = 0

print("Device:", device)
print("Loaded adapter:", ADAPTER_PATH)
print("Episodes per puzzle:", GROUP_SIZE)

for i, puzzle in enumerate(puzzles, start=1):
    print(
        f"\nPuzzle {i}/{len(puzzles)} | "
        f"{puzzle['start']} -> {puzzle['target']}"
    )

    samples = []

    for sample_index in range(GROUP_SIZE):
        episode = play_episode(
            model,
            tokenizer,
            puzzle,
            do_sample=True,
            temperature=TEMPERATURE,
        )
        samples.append(episode)

        actions = " ".join(episode["actions"])
        print(
            f"  Sample {sample_index + 1}: {actions!r} | "
            f"stop={episode['stop_reason']} | "
            f"reward={episode['reward']}"
        )

    rewards = [episode["reward"] for episode in samples]
    mixed = min(rewards) != max(rewards)
    mixed_groups += int(mixed)

    print("Rewards:", rewards, "| mixed:", mixed)

    groups.append({
        "start": puzzle["start"],
        "target": puzzle["target"],
        "rewards": rewards,
        "mixed": mixed,
        "samples": samples,
    })

report = {
    "model": MODEL,
    "revision": REVISION,
    "adapter": ADAPTER_PATH,
    "split": "train",
    "seed": SEED,
    "do_sample": True,
    "temperature": TEMPERATURE,
    "top_p": 1.0,
    "top_k": 0,
    "max_turns": MAX_TURNS,
    "max_new_tokens_per_turn": MAX_NEW_TOKENS,
    "group_size": GROUP_SIZE,
    "mixed_groups": mixed_groups,
    "total_groups": len(groups),
    "groups": groups,
}

# Include group size in the filename to preserve the previous probe.
output_path = (
    Path(ADAPTER_PATH).parent
    / f"interactive_grpo_probe_g{GROUP_SIZE}.json"
)
output_path.write_text(json.dumps(report, indent=2) + "\n")

print(f"\nGroups with mixed rewards: {mixed_groups}/{len(groups)}")
print("Saved:", output_path)