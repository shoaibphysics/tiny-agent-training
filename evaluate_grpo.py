"""Evaluate the saved GRPO adapter on held-out validation puzzles.

Use the same prompt, greedy decoding and scorer as the SFT evaluation.
Save every generated plan and its score.
"""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline import MODEL, REVISION, MAX_NEW_TOKENS
from baseline import make_prompt, generate_answer
from game import score

ADAPTER_PATH = "runs/grpo-batched-160steps/adapter"
SPLIT = "test"

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

puzzles = json.loads(Path(f"data/{SPLIT}.json").read_text())
results = []

for i, puzzle in enumerate(puzzles, start=1):
    start, target = puzzle["start"], puzzle["target"]
    prompt = make_prompt(tokenizer, start, target)
    answer = generate_answer(model, tokenizer, prompt)
    reward = score(answer, start, target)

    results.append({
        "start": start,
        "target": target,
        "answer": answer,
        "score": reward,
    })

    print(
        f"{i}/{len(puzzles)} | {start} -> {target} | "
        f"{answer!r} | score={reward}"
    )

correct = int(sum(row["score"] for row in results))
total = len(results)

report = {
    "model": MODEL,
    "revision": REVISION,
    "adapter": ADAPTER_PATH,
    "split": SPLIT,
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
    "correct": correct,
    "total": total,
    "results": results,
}

output_path = Path(ADAPTER_PATH).parent / f"{SPLIT}.json"
output_path.write_text(json.dumps(report, indent=2) + "\n")

print(
    f"\n{SPLIT.capitalize()} success: "
    f"{correct}/{total} = {correct / total:.1%}"
)
print("Saved:", output_path)