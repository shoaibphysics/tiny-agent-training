"""A tiny number-planning game.

Transform a start integer into a target using 1-3 moves:
add (+1), sub (-1), or double (*2).
score() returns 1.0 for a valid plan ending at the target,
and 0.0 otherwise. This module checks plans; it does not train a model.
"""

MOVES = {
    "add": lambda x: x + 1,
    "sub": lambda x: x - 1,
    "double": lambda x: x * 2}


def score(text, start, target):
    actions = text.split()

    if not 1 <= len(actions) <= 3:
        return 0.0

    value = start
    for action in actions:
        if action not in MOVES:
            return 0.0
        value = MOVES[action](value)

    return float(value == target)


if __name__ == "__main__":
    plan = "double double sub"
    result = score(plan, start=3, target=11)

    print("Start: 3; target: 11")
    print("Plan:", plan)
    print("Score:", result)

    assert result == 1.0
    assert score("double sub", 4, 8) == 0.0
    assert score("multiply", 3, 6) == 0.0
    assert score("", 3, 3) == 0.0
    assert score("add add add add", 3, 7) == 0.0

    print("All scorer checks passed.")