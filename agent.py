"""Better DQN-style agent scaffold for OBELIX (CPU).

This agent is *evaluation-only*: it loads pretrained weights from a file
placed next to agent.py inside the submission zip (weights.pth).

Why your STD is huge:
- if the policy is stochastic (epsilon > 0) during evaluation, scores vary a lot.
Fix:
- greedy action selection (epsilon=0), model.eval(), torch.no_grad().
- optional action smoothing to reduce oscillation when Q-values are close.

Submission ZIP structure:
  submission.zip
    agent.py
    weights.pth
"""

from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
k = 8

class DQN(nn.Module):
    def __init__(self, in_dim=18*k, n_actions=5):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,n_actions)
        )

    def forward(self, x):
        f = self.fc1(x)
        v = self.value(f)
        a = self.advantage(f)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q


_model: Optional[DQN] = None
_last_action: Optional[int] = None
_state_buffer = deque(maxlen=k)
_repeat_count: int = 0
_last_obs: Optional[np.ndarray] = None
_boundary_recovery: int = 0
_recovery_action: int = 0
_sweep_step: int = 0
_post_recovery_fw: int = 5

_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05

_COMMIT_STEPS = 3
_commit_action: Optional[int] = None
_commit_left: int = 0

FORCE_FW_PROB = 0 # Probability of forcing "FW" action
FW_BIAS = 0 # Bias added to "FW" Q-value

def _load_once():
    global _model
    if _model is not None:
        return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights_sf_10_st_650_d3qn_ep1200.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )
    m = DQN()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _boundary_recovery, _recovery_action, _sweep_step, _post_recovery_fw

    _load_once()

    if len(_state_buffer) == 0:
        for _ in range(k):
            _state_buffer.append(obs)       
    _state_buffer.append(obs)
    state = np.concatenate(_state_buffer)

    
    
    # post recovery — force FW to move away from wall
    if _post_recovery_fw > 0:
        _post_recovery_fw -= 1 
        if _post_recovery_fw == 0:
        # reset state buffer — fresh context after wall escape
            _state_buffer.clear()
            for _ in range(k):
                _state_buffer.append(obs) 
        return ACTIONS[2]

    # stuck recovery
    if _boundary_recovery > 0:
        _boundary_recovery -= 1
        return ACTIONS[_recovery_action]

    if obs[17] == 1:
        _recovery_action = 0 
        _boundary_recovery = 2
        _post_recovery_fw = 30 # after turning, force 5 FW steps
        return ACTIONS[_recovery_action]

    

    # blind — structured sweep  
    if np.sum(obs[:17]) == 0:
        _sweep_step += 1
        phase = _sweep_step % 25        
        return ACTIONS[2] if phase < 22 else ACTIONS[1]

    # sensors firing — let network decide
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    q = _model(x).squeeze(0).cpu().numpy()

    # near sensors firing — bias FW to fix flip flop
    if np.sum(obs[4:12][::2]) > 0:
        q[2] += 0.4

    return ACTIONS[int(np.argmax(q))]