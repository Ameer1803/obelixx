"""OBELIX variation: terminate immediately when enable_push becomes True.

- Episode ends once the robot attaches the box (enable_push == True).
- Terminal reward delta is +100 at that moment.
- Optimized for training speed (no rendering).
"""

from __future__ import annotations

import numpy as np
from obelix_fast import OBELIXFast
from obelix import OBELIX


class OBELIXFind(OBELIXFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fw_count = 0  # Initialize forward action count

    def update_reward(self):
        reward = 0.0

        # One-time sensor-bit bonuses (incurred only once per episode).
        sensor_bits = self.sensor_feedback[:17].astype(bool)
        weights = np.zeros(17, dtype=float)
        weights[:4] = 1.0
        weights[12:16] = 1.0
        weights[4:12][::2] = 2.0
        weights[4:12][1::2] = 3.0
        weights[16] = 5.0

        newly_on = sensor_bits & (~self._sensor_reward_claimed)
        if np.any(newly_on):
            reward += float(np.sum(weights[newly_on]))
            self._sensor_reward_claimed |= sensor_bits

        # Stuck penalty (covers wall/boundary/blocked forward cases).
        # if bool(self.sensor_feedback[17]):
        #     reward += -5.0

        # Reward for forward action count being a multiple of 5
        if self._fw_count % 5 == 0 and self._fw_count > 0:
            if not hasattr(self, '_fw_reward_claimed'):
                self._fw_reward_claimed = set()

            if self._fw_count not in self._fw_reward_claimed:
                reward += 10.0
                self._fw_reward_claimed.add(self._fw_count)

        # Continuous proximity signal — fires every step
        dist = np.linalg.norm(np.array([self.bot_center_x, self.bot_center_y]) - np.array([self.box_center_x, self.box_center_y]))
        max_dist = self.arena_size * np.sqrt(2)
        proximity_reward = (1.0 - dist / max_dist) * 0.5
        reward += proximity_reward

        reward += -1.0
        self.reward = float(reward)

    def check_done_state(self):
        # Keep existing base reward logic and push activation behavior.
        super().check_done_state()

        # If the box becomes attached, terminate immediately with +100 reward.
        if self.enable_push and not self.done:
            self.done = True
            self.reward += 2000.0

    def step(self, move, render=True):
        obs, rew, done = super().step(move, render=render)
        return obs, rew, done

    def reset(self, *args, **kwargs):
        self._fw_count = 0  # Reset forward action count at the start of each episode
        self._fw_reward_claimed = set()
        return super().reset(*args, **kwargs)
