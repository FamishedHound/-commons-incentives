"""The replay memory shared by the four landowners of one learner."""

from __future__ import annotations

import random
from collections import namedtuple

Experience = namedtuple(
    "Experience",
    ("states", "actions", "next_states", "rewards", "global_state", "next_global_state"),
)


class ReplayMemory:
    """A ring buffer of :data:`Experience`, sampled uniformly with Python's ``random``."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory: list = []
        self.position = 0

    def push(self, *args) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def push_experience(self, experience) -> None:
        """Insert an already-built transition (used by the reward-sharing learner)."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
