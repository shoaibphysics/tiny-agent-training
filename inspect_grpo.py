"""Inspect the SFT model's sampled plans before GRPO.

Sample four plans per training puzzle and score each one.
Check whether rewards differ within each group.
This script does not update model weights.
"""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from baseline import MODEL, REVISION, MAX_NEW_TOKENS, make_prompt
from game import score

ADAPTER_PATH = "runs/sft-3epochs/adapter"
NUM_PUZZLES = 12
GROUP_SIZE = 4
TEMPERATURE = 0.7
SEED = 42

set_seed(SEED)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

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

print("Device:", device)
print("Loaded adapter:", ADAPTER_PATH)

# The training file was already shuffled when we created the splits.
puzzles = json.loads(Path("data/train.json").read_text())[:NUM_PUZZLES]
groups = []
mixed_groups = 0

for i, puzzle in enumerate(puzzles, start=1):
    start, target = puzzle["start"], puzzle["target"]
    prompt = make_prompt(tokenizer, start, target)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    prompt_length = inputs["input_ids"].shape[1]

    # Sampling allows different plans for the same puzzle.
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=GROUP_SIZE,
            temperature=TEMPERATURE,
            top_p=1.0,
            top_k=0,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    print(f"\nPuzzle {i}/{len(puzzles)} | {start} -> {target}")
    samples = []

    for output in outputs:
        answer = tokenizer.decode(
            output[prompt_length:],
            skip_special_tokens=True,
        ).strip()
        reward = score(answer, start, target)

        samples.append({"answer": answer, "reward": reward})
        print(f"  {answer!r} | reward={reward}")

    rewards = [sample["reward"] for sample in samples]
    mixed = min(rewards) != max(rewards)
    mixed_groups += int(mixed)

    print("Rewards:", rewards, "| mixed:", mixed)
    groups.append({
        "start": start,
        "target": target,
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
    "max_new_tokens": MAX_NEW_TOKENS,
    "group_size": GROUP_SIZE,
    "mixed_groups": mixed_groups,
    "total_groups": len(groups),
    "groups": groups,
}

output_path = Path("results/grpo_probe.json")
output_path.parent.mkdir(exist_ok=True)
output_path.write_text(json.dumps(report, indent=2) + "\n")

print(f"\nGroups with mixed rewards: {mixed_groups}/{len(groups)}")
print("Saved:", output_path)