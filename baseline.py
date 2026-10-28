"""Measure the original model on the selected number-game split.

Generate one plan per puzzle, score it and save the answers.
No model weights are updated.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from game import score

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MAX_NEW_TOKENS = 32
SPLIT = "test"


def make_prompt(tokenizer, start, target):
    question = (
        f"Start at {start}. Finish at {target} using one to three moves. "
        "add means +1; sub means -1; double means ×2. "
        "Return only the move names, separated by spaces."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.float32,
    ).to(device)
    model.eval()

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
        "split": SPLIT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "correct": correct,
        "total": total,
        "results": results,
    }

    output_path = Path(f"results/baseline_{SPLIT}.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{SPLIT.capitalize()} success: {correct}/{total} = {correct / total:.1%}")
    print("Saved:", output_path)