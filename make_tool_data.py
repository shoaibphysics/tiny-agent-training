"""Convert training plans into verified tool-use demonstrations.

Each row teaches one assistant response given the preceding conversation.
Environment observations come from Python. DONE teaches when to stop.
This prepares data only; it does not train the model.
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

# Preserve the original split: use training puzzles only.
puzzles = json.loads(Path("data/train.json").read_text())
rows = []
longest_example = 0

for puzzle in puzzles:
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

    # None represents the final DONE response, not another game move.
    for action in puzzle["completion"].split() + [None]:
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        if action is None:
            assert env.get_reward() == 1.0
            completion = "DONE" + tokenizer.eos_token
        else:
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
        longest_example = max(longest_example, len(input_ids))

        rows.append({
            "start": puzzle["start"],
            "target": puzzle["target"],
            "turn": len(env.actions) + 1,
            "prompt": prompt,
            "completion": completion,
        })

        if action is None:
            break

        assistant = parse_response(
            tokenizer,
            completion_ids,
            prefix=prompt_ids,
            tools=tools,
        )
        calls = assistant.get("tool_calls") or []
        assert len(calls) == 1
        assert calls[0]["function"] == request

        parsed_action = calls[0]["function"]["arguments"]["action"]
        observation = env.move(parsed_action)

        messages.extend([
            assistant,
            {"role": "tool", "name": "move", "content": observation},
        ])

# Check that a 512-token training limit will preserve every target.
assert longest_example <= 512

output_path = Path("data/tool_train.json")
output_path.write_text(json.dumps(rows, indent=2) + "\n")

print("Source training puzzles:", len(puzzles))
print("Tool-call examples:", len(rows) - len(puzzles))
print("DONE examples:", len(puzzles))
print("Total training rows:", len(rows))
print("Longest example in tokens:", longest_example)
print("Saved:", output_path)