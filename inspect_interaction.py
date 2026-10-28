"""Inspect one model-driven interaction after tool-use SFT.

The model generates requests; Python executes them.
Real environment observations become context for subsequent decisions.
This uses a training puzzle as an interface check, not a generalization test.
"""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.chat_template_utils import add_response_schema, parse_response

from baseline import MODEL, REVISION
from environment import NumberGame

ADAPTER_PATH = "runs/tool-sft-1epoch/adapter"

# Three action responses plus a possible final DONE response.
# The environment still permits at most three game moves.
MAX_TURNS = 4
MAX_NEW_TOKENS = 128

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
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

puzzle = json.loads(Path("data/train.json").read_text())[0]

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

raw_replies = []
protocol_error = False

print("Device:", device)
print("Loaded adapter:", ADAPTER_PATH)
print("Initial observation:", observation)

for turn in range(MAX_TURNS):
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_ids = inputs["input_ids"][0].tolist()
    new_ids = output[0, len(prompt_ids):].tolist()
    raw_reply = tokenizer.decode(new_ids, skip_special_tokens=False)

    raw_replies.append(raw_reply)
    print(f"\nModel turn {turn + 1}:\n{raw_reply}")

    if not new_ids or new_ids[-1] != tokenizer.eos_token_id:
        print("Incomplete reply: generation reached its token limit.")
        protocol_error = True
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
        if "<tool_call" in raw_reply:
            print("Malformed tool request; no action executed.")
            protocol_error = True
        else:
            print("No tool call: the model ended the episode.")
        break

    print("Tool calls in this reply:", len(calls))

    # Execute every requested call rather than silently selecting one.
    for call in calls:
        function = call.get("function", {})
        arguments = function.get("arguments")

        if (
            function.get("name") != "move"
            or not isinstance(arguments, dict)
            or set(arguments) != {"action"}
            or not isinstance(arguments["action"], str)
        ):
            print("Invalid tool request:", function)
            protocol_error = True
            break

        observation = env.move(arguments["action"])
        print("Environment:", observation)

        messages.append({
            "role": "tool",
            "name": "move",
            "content": observation,
        })

    if protocol_error or env.invalid:
        break

reward = 0.0 if protocol_error else env.get_reward()

report = {
    "model": MODEL,
    "revision": REVISION,
    "adapter": ADAPTER_PATH,
    "start": puzzle["start"],
    "target": puzzle["target"],
    "max_turns": MAX_TURNS,
    "max_new_tokens_per_turn": MAX_NEW_TOKENS,
    "do_sample": False,
    "messages": messages,
    "raw_replies": raw_replies,
    "actions": env.actions,
    "final_value": env.value,
    "protocol_error": protocol_error,
    "reward": reward,
}

# Save beside this adapter, preserving the earlier failed probe.
output_path = Path(ADAPTER_PATH).parent / "interactive_probe.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, indent=2) + "\n")

print("\nExecuted actions:", env.actions)
print("Final value:", env.value)
print("Final reward:", reward)
print("Saved:", output_path)