"""Generate verified number-game examples and fixed data splits.

Each example contains a start, target and successful action plan.
Distinct puzzles are separated into training, validation and test sets.
"""

import json
from itertools import product
from pathlib import Path
from random import Random

from game import MOVES, score


def make_examples():
    examples = {}

    for start in range(1, 21):
        for length in range(1, 4):
            for actions in product(MOVES, repeat=length):
                value = start
                for action in actions:
                    value = MOVES[action](value)

                key = (start, value)
                if value != start and key not in examples:
                    examples[key] = {
                        "start": start,
                        "target": value,
                        "completion": " ".join(actions),
                    }

    return list(examples.values())


if __name__ == "__main__":
    examples = make_examples()

    assert all(
        score(row["completion"], row["start"], row["target"]) == 1.0
        for row in examples
    )

    Random(42).shuffle(examples)
    n = len(examples)
    train_end = int(0.8 * n)
    validation_end = int(0.9 * n)

    splits = {
        "train": examples[:train_end],
        "validation": examples[train_end:validation_end],
        "test": examples[validation_end:],
    }

    task_sets = [
        {(row["start"], row["target"]) for row in rows}
        for rows in splits.values()
    ]
    train, validation, test = task_sets
    assert not (train & validation or train & test or validation & test)

    output = Path("data")
    output.mkdir(exist_ok=True)

    for name, rows in splits.items():
        path = output / f"{name}.json"
        path.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"{name}: {len(rows)} examples")

    print("First training example:", splits["train"][0])