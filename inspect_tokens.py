"""Inspect one SFT example before training.

The prompt provides context. Only the answer and its ending token
contribute directly to the training loss.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

from baseline import MODEL, REVISION, make_prompt

tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)

example = json.loads(Path("data/train.json").read_text())[0]
prompt = make_prompt(tokenizer, example["start"], example["target"])
completion = example["completion"] + tokenizer.eos_token

prompt_ids = tokenizer(
    prompt, add_special_tokens=False
)["input_ids"]

input_ids = tokenizer(
    prompt + completion, add_special_tokens=False
)["input_ids"]

# Check that joining the answer did not change the prompt's tokenization.
n_prompt = len(prompt_ids)
assert input_ids[:n_prompt] == prompt_ids

# -100 tells the loss function to ignore a target position.
labels = [-100] * n_prompt + input_ids[n_prompt:]
loss_mask = [int(label != -100) for label in labels]

print("Example:", example)
print("\nFull training text:")
print(prompt + completion)

print("\nPrompt tokens:", n_prompt)
print("Total tokens:", len(input_ids))
print("Supervised tokens:", sum(loss_mask))

print("\npos | token ID | token text             | label   | loss")
for i, (token_id, label, mask) in enumerate(
    zip(input_ids, labels, loss_mask)
):
    piece = tokenizer.decode([token_id])
    print(f"{i:3} | {token_id:8} | {piece!r:22} | {label:7} | {mask}")

print("\nLabels:", labels)
print("Loss mask:", loss_mask)