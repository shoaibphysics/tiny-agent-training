"""An interactive version of the number game.

The environment stores the current number and action history.
Each move returns an observation for the next decision.
The existing scorer determines success when the episode finishes.
"""

from game import MOVES, score


class NumberGame:
    def reset(self, start, target, **kwargs):
        self.start = start
        self.value = start
        self.target = target
        self.actions = []
        self.invalid = False
        return self._observation()

    def _observation(self):
        return (
            f"Value: {self.value}; target: {self.target}; "
            f"moves left: {3 - len(self.actions)}"
        )

    def move(self, action: str) -> str:
        """Apply exactly one move to the current number.

        Args:
            action: One of add, sub, or double.

        Returns:
            The current number, target, and remaining move budget.
        """
        if len(self.actions) >= 3:
            self.invalid = True
            return "Move limit exceeded; reward is zero. " + self._observation()

        action = action.strip()
        self.actions.append(action)

        if action not in MOVES:
            self.invalid = True
            return "Invalid action; one move consumed. " + self._observation()

        self.value = MOVES[action](self.value)
        return self._observation()

    def get_reward(self) -> float:
        if self.invalid:
            return 0.0
        return score(" ".join(self.actions), self.start, self.target)


if __name__ == "__main__":
    env = NumberGame()

    print(env.reset(start=3, target=11))
    for action in ["double", "double", "sub"]:
        print(f"{action}: {env.move(action)}")
    print("Final reward:", env.get_reward())
    assert env.get_reward() == 1.0

    # A fourth attempt cannot change the number and invalidates the episode.
    env.move("add")
    assert env.value == 11
    assert env.get_reward() == 0.0

    # Reaching the target does not prevent a later mistake.
    env.reset(start=4, target=8)
    env.move("double")
    assert env.get_reward() == 1.0
    env.move("sub")
    assert env.get_reward() == 0.0

    # One tool call must contain exactly one action.
    env.reset(start=3, target=11)
    env.move("double double sub")
    assert len(env.actions) == 1
    assert env.get_reward() == 0.0

    print("Environment checks passed.")