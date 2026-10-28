# Tiny Agent Training

Learning LLM post-training through one small number game, from complete plans to an agent that calls a tool and observes its result.

The project was built incrementally:

1. **Whole-plan SFT:** learn verified action sequences.
2. **Whole-plan GRPO:** sample plans and learn from their executable rewards.
3. **Tool-use SFT:** learn to request one action, read the observation, and stop.
4. **Interactive GRPO:** learn from the rewards of complete tool-use episodes.

After the first SFT run, each stage continues the preceding adapter. Teacher-to-student distillation remains a future experiment.

## Task, data, and model

Given a start integer and a target, reach the target using one to three actions:

- `add`: add one
- `sub`: subtract one
- `double`: multiply by two

For example, `3 → 11` has the plan `double double sub`: `3 → 6 → 12 → 11`.

The scorer executes every action. It returns `1.0` only when a legal, nonempty plan ends exactly on the target within three moves; otherwise it returns `0.0`. Reaching the target and then moving away fails. Any successful legal plan is accepted, not just the reference answer. There is no partial reward.

A solver enumerates all 39 sequences of lengths one through three for starting values 1–20. It keeps one shortest solution per distinct `(start, target)` pair and excludes targets equal to their start. Shuffling the 418 puzzles with seed 42 gives:

| Training | Validation | Test |
|---:|---:|---:|
| 334 | 42 | 42 |

Puzzle pairs are disjoint across splits; numbers, action patterns, and intermediate states can overlap. Tool demonstrations and GRPO prompts use training puzzles only.

The original checkpoint is `Qwen/Qwen3-0.6B`, pinned to this Hugging Face commit:

```text
c1899de289a04d12100db370d81485cdf75e47ca
```

All stages freeze the original weights and update LoRA parameters:

| Setting | Value |
|---|---|
| LoRA rank / alpha | 8 / 16 |
| Target modules | `q_proj`, `v_proj` |
| Trainable parameters | 1,146,880, approximately 0.192% |
| Precision / device used | Float32 / Apple Silicon MPS |
| Seed | 42 |

“Original model” means the existing Qwen checkpoint before this project's additional training.

## 1. Whole-plan SFT

The first interface asks for an entire plan as whitespace-separated action names. SFT learns the solver's verified demonstrations using next-token cross-entropy. The scorer checks answers but does not drive this loss.

Before training, I inspected a tokenized example and the trainer's actual labels. For `12 → 96`, the chat template supplies this **54-token context**:

```text
<|im_start|>user
Start at 12. Finish at 96 using one to three moves. add means +1; sub means -1; double means ×2. Return only the move names, separated by spaces.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The assistant's opening marker is already supplied. With thinking disabled, this template also supplies the empty thinking block. The **target completion** is:

```text
double double double<|im_end|>
```

The user's `<|im_end|>` belongs to the context; the assistant's `<|im_end|>` is part of the target the model learns to predict.

| Positions (zero-based) | Content | Labels | Direct loss |
|---|---|---|---|
| 0–53 | User message and assistant prefix | `-100` | No |
| 54–56 | `double`, ` double`, ` double` | `4331, 1990, 1990` | Yes |
| 57 | Assistant message ending | `151645` | Yes |

The different IDs for `double` and ` double` reflect the leading space. Only the context's labels become `-100`; its input token IDs remain unchanged.

**During training**, the input contains all 58 tokens. Teacher forcing supplies the correct preceding answer tokens, and causal attention prevents predictions from using future tokens. The loss applies only to the four completion targets. The model handles the next-token shift internally: logits at position 53 predict the first answer token at position 54.

Masking context labels does not hide the context from attention.

**During inference**, only the 54-token context is supplied. The model generates its own completion one token at a time, including the ending marker; its answer can still be wrong.

Training used three epochs over 334 examples, batch size 2, no gradient accumulation, and a constant learning rate of `1e-4`: 501 optimizer updates, approximately 88 seconds.

## 2. Whole-plan GRPO

GRPO starts from the SFT adapter, samples its own plans, executes them, and learns from binary rewards. It does not use reference answers or a learned reward model.

Each collection contains four training puzzles with eight sampled plans each. Rewards are normalized **within each puzzle's group**. Plans are reused for two optimization passes, then fresh plans are collected from the updated policy. Old-policy log probabilities stay fixed during reuse; clipping limits the objective's incentive for large probability-ratio changes.

| Setting | Whole-plan GRPO |
|---|---|
| Collection / optimizer batch | 32 plans / 8 plans |
| Microbatch / accumulation | 4 / 2 |
| Passes per collection | 2 |
| Collections / optimizer updates | 20 / 160 |
| Fresh plans / distinct training puzzles | 640 / 80 |
| Learning rate | Constant `1e-5` |
| Temperature / top-p / top-k | 0.7 / 1.0 / disabled |
| Completion budget | 32 tokens |
| Loss / clipping epsilon / KL coefficient | `grpo` / 0.2 / 0 |

Groups with identical rewards have zero relative advantages. A failed plan can have a negative advantage when other plans in its group succeed. This run took approximately 114 seconds.

### Whole-plan results

These evaluations use matched prompts, greedy decoding, thinking disabled, 32 generated tokens maximum, and the same scorer.

| Checkpoint | Validation | Test | Valid test format |
|---|---:|---:|---:|
| Original Qwen | 0/42 — 0.0% | 0/42 — 0.0% | 0/42 |
| SFT | 12/42 — 28.6% | 20/42 — 47.6% | 42/42 |
| SFT → GRPO | 17/42 — 40.5% | 22/42 — 52.4% | 42/42 |

GRPO added five validation successes without regressions. On test it added four successes and lost two, for a net gain of two. For example, `10 → 40` regressed from `double double` to `double`. Aggregate improvement did not preserve every previous success.

## 3. An environment and tool-use SFT

The interactive interface changes the execution loop:

```text
model generates request → Python parses and checks it → environment executes
                     ↑                                  ↓
                updated history ← returned observation
```

`NumberGame.reset` initializes state; `move` executes an action; `get_reward` checks the recorded episode. Observations contain the current number, target, and moves remaining. The environment enforces the three-move budget. Invalid actions consume a move and invalidate the episode; a fourth action also invalidates it.

Passing `tools=[env.move]` supplies a function description to the chat template. It does not execute Python. The model predicts the tool markers, function name, JSON arguments, and message ending. The parser recognizes that request, then Python calls `env.move(action)` and appends the result before the next generation.

The whole-plan GRPO model initially produced ordinary text under this interface, with zero interactive validation successes. Tool-use SFT teaches the new response format and decision behavior.

### A complete action–observation trajectory

The tool-SFT model produced this successful `12 → 96` episode. Role labels below are for readability; the system tool schema and chat-template headers are omitted.

```text
User: Play the number game using the move tool.
      Value: 12; target: 96; moves left: 3

Assistant:
<tool_call>
{"name": "move", "arguments": {"action": "double"}}
</tool_call><|im_end|>

Tool: Value: 24; target: 96; moves left: 2

Assistant:
<tool_call>
{"name": "move", "arguments": {"action": "double"}}
</tool_call><|im_end|>

Tool: Value: 48; target: 96; moves left: 1

Assistant:
<tool_call>
{"name": "move", "arguments": {"action": "double"}}
</tool_call><|im_end|>

Tool: Value: 96; target: 96; moves left: 0

Assistant: DONE<|im_end|>

Final reward: 1.0
```

There are three actions but four assistant responses. `DONE` is a learned stopping response, not an arithmetic action. Saying it at the wrong value still earns zero.

### What is supervised?

`make_tool_data.py` converts correct training plans into 1,143 decision rows: 809 tool requests and 334 `DONE` responses. It constructs reference requests and executes them to obtain real observations; it does not ask a model to generate the demonstrations.

A three-move puzzle becomes four rows:

| Row | Context visible to the model | Supervised response |
|---|---|---|
| 1 | Instructions and initial observation | First tool request |
| 2 | Above, plus first request and result | Second tool request |
| 3 | Above, plus second request and result | Third tool request |
| 4 | Above, plus third request and result | `DONE` |

Every row retains the full preceding conversation. Earlier calls are targets in their own rows and masked context in later rows. Masking removes their direct loss contribution, not their visibility to attention.

For the third `double` request in the inspected example, the actual template produced **381 context tokens and 19 target tokens**. The target is the tool request shown above, including its ending token:

```text
Target IDs:
[151657, 198, 4913, 606, 788, 330, 3397, 497, 330,
 16370, 788, 5212, 1311, 788, 330, 4331, 95642, 151658, 151645]

labels    = [-100] × 381 + target IDs
loss mask = [0]    × 381 + [1] × 19
```

`<tool_call>` is ID 151657, `</tool_call>` is 151658, and `<|im_end|>` is 151645. These entries already exist in the tokenizer; the project adds no vocabulary. The tool markers are dedicated vocabulary entries, although their saved metadata does not mark them as tokenizer-special tokens.

The longest training row is 437 tokens, below the 512-token limit. One epoch continued the whole-plan GRPO adapter for 572 updates, batch size 2, constant learning rate `1e-4`, approximately 486 seconds. This is SFT because it minimizes demonstration cross-entropy, regardless of the starting checkpoint's GRPO history.

## 4. Interactive GRPO

Before training, a probe sampled eight complete episodes for each of 12 training puzzles. Four groups contained both successes and failures, establishing that some group-relative reward signal was available.

`train_interactive_grpo.py` continues the tool-SFT adapter. TRL's experimental `environment_factory` creates an independent game for each episode. The model requests actions, receives observations, and ultimately receives a terminal binary reward. A protocol guard cancels rewards for invalid traces, including an unexecuted final tool request; it does not award a second success reward.

The collection schedule remains four puzzles × eight episodes, with two passes and eight updates per collection. Microbatch size is now 2 with accumulation 4. The full run used 160 updates, 640 fresh episodes across 80 training puzzles, and 1,280 episode training uses. Learning rate, sampling, clipping, and KL settings match whole-plan GRPO. The trajectory budget is 768 tokens including inserted context, with 128 new tokens per response.

The policy objective applies to **model-generated tokens across the episode**, including tool requests and stopping tokens. Environment observations and inserted conversation headers remain context but are excluded from policy loss. In the inspected three-action smoke example:

| Tokens after the initial prompt | Count | Policy-loss eligibility |
|---|---:|---|
| Three generated requests and `DONE` | 59 | Included |
| Environment responses and inserted headers | 102 | Excluded |

The same episode advantage weights its generated decisions. This differs from per-turn SFT, which supplies a verified next response. The smoke test also showed zero reported loss with a nonzero gradient: zero scalar loss does not imply that its derivatives vanish.

Full training took approximately 977 seconds. Thirty of 80 puzzle groups had mixed rewards. The 366/640 successful training episodes describe changing policies and prompts, not the final checkpoint's held-out accuracy.

### Interactive results

Matched evaluations use greedy decoding, thinking disabled, at most four assistant turns, 128 new tokens per turn, and the same prompt, parser, environment, and scorer.

| Checkpoint | Validation | Test |
|---|---:|---:|
| Whole-plan GRPO, before tool SFT | 0/42 — 0.0% | Not evaluated |
| Tool SFT | 21/42 — 50.0% | 23/42 — 54.8% |
| Tool SFT → interactive GRPO | 27/42 — 64.3% | 32/42 — 76.2% |

Both interactive-trained checkpoints stopped with `DONE` on all 42 puzzles in each split, with no invalid tool calls or game actions. Validation had 12 gains and six regressions; test had ten gains and one regression. The net improvements were 14.3 and 21.4 percentage points, respectively.

Results by **minimum solution length**, reported as tool SFT → interactive GRPO:

| Minimum moves | Validation | Test |
|---|---:|---:|
| 1 | 8/8 → 8/8 | 5/5 → 5/5 |
| 2 | 3/10 → 10/10 | 12/19 → 18/19 |
| 3 | 10/24 → 9/24 | 6/18 → 9/18 |

These buckets describe the shortest solution, not the generated plan's length; any legal successful plan within three moves earns credit. Test also contains fewer three-move puzzles than validation, so the two headline percentages have different puzzle mixes.

The three-move result differs across splits. On validation, all six regressions involved action ordering: for `18 → 40`, `add add double` succeeds but `double add add` ends at 38. The test regression was `19 → 42`: `add add double` became `double add double`, ending at 78. Valid syntax and stopping do not guarantee correct planning.

## What the experiments taught me

### Token prediction and solving a puzzle measure different things

Tool-SFT training logs reported token accuracy around 99.38%, but the resulting model solved 21/42 validation puzzles. These measurements differ in both data and evaluation conditions: training token accuracy measures individual next-token predictions under reference histories, whereas validation success requires a complete generated episode to reach the target.

Most tokens in a tool request express repeated JSON syntax, tool names, and message boundaries. Predicting those correctly does not guarantee choosing the right action. During SFT, the model receives correct preceding tokens and demonstrated histories; during evaluation, it must continue from its own decisions and their consequences.

A related concept we discussed is that next-token loss can decrease without changing greedy success: the correct token's probability can rise while an incorrect token remains the most probable. We did not measure a validation-loss curve demonstrating this effect in these runs.

### Valid tool use is necessary, but it is not planning success

Before tool SFT, the whole-plan GRPO model produced ordinary text and executed no actions on validation. After tool SFT, it made valid requests and stopped with `DONE` on all 42 puzzles, but solved only half.

After interactive GRPO, protocol validity and stopping remained intact while validation and test success increased. The additional successes therefore reflect the chosen action sequences rather than simply more valid formatting.

`DONE` ends the interaction. It does not certify that the target was reached; the executable scorer makes that decision.

### A zero GRPO loss does not necessarily mean no learning

The interactive smoke test reported loss 0 with a gradient norm of approximately 0.2405, and all 112 adapter tensors changed. At the initial update, importance ratios were one and the centered advantages averaged to zero, so contributions to the scalar objective cancelled. Their derivatives need not cancel, because different sampled decisions depend differently on the parameters.

I therefore checked the actual gradients and saved weights instead of interpreting the loss value alone as evidence that training had failed. GRPO loss is not a percentage of puzzles solved and need not resemble a positive cross-entropy learning curve.

### Reward variation must exist within the same puzzle

One collection contained 32 episodes and eight successes, yet provided no group-relative reward signal. Its four puzzle groups were individually constant: three groups had eight failures each, and one had eight successes.

There was reward variation across the collection, but none within a puzzle's group. All relative advantages were zero. The corresponding log showed zero loss and gradient norm in this configuration, which had no KL term. A zero current gradient does not necessarily imply unchanged parameters when optimizer momentum exists.

This explains why collecting more episodes and taking more optimizer steps are different choices. Reusing an all-failure group does not reveal which sampled plan is better. Fresh sampling may discover a successful alternative, but a larger group does not guarantee that it will.

### Aggregate improvement can hide action-order regressions

Interactive GRPO produced 12 validation gains and six regressions, and ten test gains with one regression. The test regression, `19 → 42`, changed from:

- `add add double`: 19 → 20 → 21 → 42.
- `double add double`: 19 → 38 → 39 → 78.

Both plans are syntactically valid, but ordering changes their meaning. Looking at individual traces exposed a weakness that the headline score concealed. These regressions do not, by themselves, establish catastrophic forgetting; they show that improving the overall policy did not preserve every previous success.

### The second split changed my interpretation of longer plans

Initially, validation suggested that gains were concentrated in two-move puzzles: 3/10 → 10/10, while three-move performance changed 10/24 → 9/24. A plausible explanation was that longer plans require more coordinated choices. Three actions yield nine sequences of exactly two moves and 27 of exactly three.

However, three-move test performance improved from 6/18 to 9/18. Longer plans remained harder in these evaluations, but GRPO did improve some of them. Sequence counts do not establish how much data is required, and longer training is a hypothesis to test, not a guaranteed remedy.

Before changing training duration, I would inspect training exposure, sampled successes, and mixed-reward groups by minimum solution length. That would help distinguish insufficient coverage or successful exploration from insufficient optimization. This analysis by length is a proposed next check, not an experiment already performed.

## Limitations

These are single-run results on small fixed splits from one generated domain. Puzzle pairs are held out, but numbers, action patterns, and intermediate states can overlap. The results do not establish reliable gains across seeds, larger-number generalization, or general planning ability. Minimum solution length also differs from generated length, and validation and test contain different proportions of puzzle lengths.

The test puzzles were already evaluated in earlier whole-plan experiments. They remain excluded from training; they are not a never-inspected benchmark. Further training decisions should use validation rather than repeated tuning against these test results.

The project implements a real action–observation loop, but no ablation establishes that using observations caused the gains. Interface and training data also changed between whole-plan and interactive stages.

Native GRPO preserves generated token history internally; the manual evaluator rerenders the original chat template. Both compared checkpoints use the same evaluator/template, but training and evaluation history formatting are not identical.

## Reproduce

Use Python 3.12 and `uv`; run from the repository root. `uv.lock` records dependencies. The scripts use fixed run paths; preserve previous outputs before intentionally rerunning training.

The GitHub repository includes code, fixed puzzle splits, and JSON experiment evidence; model adapters and tokenizer copies are generated locally and are not included. On a fresh clone, run the training stages in order before their adapter evaluations. `data/tool_train.json` is also generated locally by `make_tool_data.py`.

```bash
uv sync --locked
uv run game.py
uv run make_data.py
uv run inspect_tokens.py
uv run baseline.py
uv run train_sft.py
uv run evaluate_sft.py
uv run inspect_grpo.py
uv run train_grpo.py
uv run evaluate_grpo.py
```

For whole-plan evaluation, set `SPLIT` to `"validation"` or `"test"` in `baseline.py`, `evaluate_sft.py`, and `evaluate_grpo.py`. Reports are saved separately.

Then build and train the interactive interface:

```bash
uv run environment.py
uv run inspect_tool_example.py
uv run make_tool_data.py
uv run train_tool_sft.py
uv run inspect_interaction.py
uv run evaluate_interaction.py --adapter runs/tool-sft-1epoch/adapter --split validation
uv run inspect_interactive_grpo.py
uv run train_interactive_grpo.py
uv run evaluate_interaction.py --adapter runs/interactive-grpo-batched-160steps/adapter --split validation
```

Evaluate the fixed interactive checkpoints on test:

```bash
uv run evaluate_interaction.py --adapter runs/tool-sft-1epoch/adapter --split test
uv run evaluate_interaction.py --adapter runs/interactive-grpo-batched-160steps/adapter --split test
```

Token inspection and one-update smoke tests preceded the recorded full runs. Current training scripts contain the full-run configurations.

## Files and saved evidence

| Files | Purpose |
|---|---|
| `game.py`, `make_data.py` | Rules, scorer, demonstrations, fixed splits |
| `baseline.py`, `inspect_tokens.py` | Original model and token-mask inspection |
| `train_sft.py`, `evaluate_sft.py` | Whole-plan SFT |
| `inspect_grpo.py`, `train_grpo.py`, `evaluate_grpo.py` | Whole-plan reward training |
| `environment.py`, `inspect_interaction.py` | State and model-driven tool loop |
| `inspect_tool_example.py`, `make_tool_data.py`, `train_tool_sft.py` | Verified tool demonstrations and SFT |
| `inspect_interactive_grpo.py`, `train_interactive_grpo.py` | Sampled episodes and interactive reward training |
| `evaluate_interaction.py` | Shared episode helper and split-aware evaluation |

Recorded stages live in `runs/sft-3epochs`, `runs/grpo-batched-160steps`, `runs/tool-sft-1epoch`, and `runs/interactive-grpo-batched-160steps`. Git tracks the JSON records directly inside each run directory: training metrics, trainer state, probes, evaluation reports, and sampled episodes. Interactive evaluations save full conversations in `interactive_validation.json` and `interactive_test.json`; interactive GRPO also saves `rollouts.json`. Adapters, tokenizer copies, and checkpoint subdirectories stay local under `.gitignore`.

## References

- [Hugging Face agent-training course materials](https://github.com/burtenshaw/training-agents)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [TRL supervised fine-tuning](https://huggingface.co/docs/trl/sft_trainer)
- [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)
- [PEFT LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora)

**AI assistance:** Most of the code was generated with AI assistance. I directed the experimental design and implementation, ran and evaluated the experiments, and take responsibility for the code, reported results, and conclusions.