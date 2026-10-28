"""Evaluate interactive game play on held-out validation or test puzzles.

The reusable episode function also supports sampled rollouts.
Save complete conversations so failures can be inspected afterward.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.chat_template_utils import add_response_schema, parse_response

from baseline import MODEL, REVISION
from environment import NumberGame

SPLIT = "validation"
MAX_TURNS = 4
MAX_NEW_TOKENS = 128


def play_episode(
    model,
    tokenizer,
    puzzle,
    *,
    do_sample=False,
    temperature=0.7,
):
    env = NumberGame()
    observation = env.reset(puzzle["start"], puzzle["target"])
    tools = [env.move]

    messages = [{
        "role": "user",
        "content": (
            "Play the number game using the move tool. "
            "add means +1; sub means -1; double means multiply by 2. "
            "Use one to three moves to finish exactly at the target. "
            "Call one move at a time and read its observation. "
            "When finished, reply DONE without calling a tool. "
            "Plain text does not execute actions. "
            "Invalid actions or exceeding the move budget score zero.\n"
            + observation
        ),
    }]

    # Sampling changes how tokens are selected, not the game rules.
    sampling_options = {}
    if do_sample:
        sampling_options = {
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 0,
        }

    raw_replies = []
    protocol_error = False
    stop_reason = "turn_limit"

    for _ in range(MAX_TURNS):
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **sampling_options,
            )

        prompt_ids = inputs["input_ids"][0].tolist()
        new_ids = output[0, len(prompt_ids):].tolist()
        raw = tokenizer.decode(new_ids, skip_special_tokens=False)
        raw_replies.append(raw)

        if not new_ids or new_ids[-1] != tokenizer.eos_token_id:
            protocol_error = True
            stop_reason = "incomplete_response"
            break

        assistant = parse_response(
            tokenizer,
            new_ids,
            prefix=prompt_ids,
            tools=tools,
        )
        messages.append(assistant)
        calls = assistant.get("tool_calls") or []

        if not calls:
            if "<tool_call" in raw:
                protocol_error = True
                stop_reason = "malformed_tool"
            else:
                stop_reason = (
                    "DONE"
                    if assistant.get("content", "").strip() == "DONE"
                    else "other_text"
                )
            break

        # Execute every requested call; the environment enforces its budget.
        for call in calls:
            function = call.get("function", {})
            arguments = function.get("arguments")

            if (
                function.get("name") != "move"
                or not isinstance(arguments, dict)
                or set(arguments) != {"action"}
                or not isinstance(arguments["action"], str)
            ):
                protocol_error = True
                stop_reason = "invalid_tool"
                break

            observation = env.move(arguments["action"])
            messages.append({
                "role": "tool",
                "name": "move",
                "content": observation,
            })

        if protocol_error or env.invalid:
            if env.invalid:
                stop_reason = "invalid_move"
            break

    return {
        "start": puzzle["start"],
        "target": puzzle["target"],
        "actions": env.actions,
        "final_value": env.value,
        "reward": 0.0 if protocol_error else env.get_reward(),
        "stop_reason": stop_reason,
        "protocol_error": protocol_error,
        "invalid": env.invalid,
        "messages": messages,
        "raw_replies": raw_replies,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default=SPLIT,
    )
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter,
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
        args.adapter,
        is_trainable=False,
    )
    model.eval()

    puzzles = json.loads(Path(f"data/{args.split}.json").read_text())
    results = []

    print("Device:", device)
    print("Loaded adapter:", args.adapter)

    for i, puzzle in enumerate(puzzles, start=1):
        result = play_episode(model, tokenizer, puzzle)
        results.append(result)
        actions = " ".join(result["actions"])

        print(
            f"{i}/{len(puzzles)} | "
            f"{puzzle['start']} -> {puzzle['target']} | "
            f"{actions!r} | stop={result['stop_reason']} | "
            f"reward={result['reward']}"
        )

    correct = int(sum(row["reward"] for row in results))
    total = len(results)
    done_count = sum(row["stop_reason"] == "DONE" for row in results)

    report = {
        "model": MODEL,
        "revision": REVISION,
        "adapter": args.adapter,
        "split": args.split,
        "max_turns": MAX_TURNS,
        "max_new_tokens_per_turn": MAX_NEW_TOKENS,
        "do_sample": False,
        "correct": correct,
        "total": total,
        "stopped_with_done": done_count,
        "results": results,
    }

    output_path = (
        Path(args.adapter).parent / f"interactive_{args.split}.json"
    )
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"\nInteractive {args.split} success: "
        f"{correct}/{total} = {correct / total:.1%}"
    )
    print(f"Stopped with DONE: {done_count}/{total}")
    print("Saved:", output_path)