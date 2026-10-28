"""Inspect a verified tool-use demonstration before training.

Correct actions come from our existing training data.
Python executes each tool request and supplies the real observation.
This script loads only the tokenizer; it does not train a model.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer
from trl.chat_template_utils import add_response_schema, parse_response

from environment import NumberGame

ADAPTER_PATH = "runs/grpo-batched-160steps/adapter"

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    local_files_only=True,
)
tokenizer = add_response_schema(tokenizer)

puzzle = json.loads(Path("data/train.json").read_text())[0]

env = NumberGame()
observation = env.reset(puzzle["start"], puzzle["target"])
tools = [env.move]

# Match the instructions used in inspect_interaction.py.
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

print("Initial observation:", observation)

for action in puzzle["completion"].split():
    # The context contains all earlier requests and observations.
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    request = {"name": "move", "arguments": {"action": action}}
    completion = (
        "<tool_call>\n"
        + json.dumps(request)
        + "\n</tool_call>"
        + tokenizer.eos_token
    )

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(
        completion, add_special_tokens=False
    )
    input_ids = tokenizer.encode(
        prompt + completion, add_special_tokens=False
    )
    assert input_ids == prompt_ids + completion_ids

    # Check that the same parser used during inference understands it.
    assistant = parse_response(
        tokenizer,
        completion_ids,
        prefix=prompt_ids,
        tools=tools,
    )
    calls = assistant.get("tool_calls") or []
    assert len(calls) == 1
    assert calls[0]["function"] == request

    # Execute the parsed request, then record the real observation.
    parsed_action = calls[0]["function"]["arguments"]["action"]
    observation = env.move(parsed_action)

    messages.extend([
        assistant,
        {"role": "tool", "name": "move", "content": observation},
    ])
    print(f"{parsed_action} -> {observation}")

assert env.get_reward() == 1.0
print("Reference demonstration reward:", env.get_reward())

# Inspect the final action's training example.
# All preceding conversation is context; only this request is supervised.
labels = [-100] * len(prompt_ids) + completion_ids

print("\nContext for the final action:\n", prompt)
print("\nTarget completion:\n", completion)
print("\nIgnored context tokens:", len(prompt_ids))
print("Supervised tokens:", len(completion_ids))

print("\npos | token ID | token text             | label | loss")
for position in range(len(prompt_ids), len(input_ids)):
    token_id = input_ids[position]
    piece = tokenizer.decode([token_id])
    print(
        f"{position:3} | {token_id:8} | {piece!r:22} | "
        f"{labels[position]:5} | 1"
    )