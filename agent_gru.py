from __future__ import annotations
from typing import Optional, List
import os
import numpy as np
import torch
import torch.nn as nn

ACTS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
DIM_IN = 18
DIM_ACT = 5


class Net(nn.Module):
    def __init__(self, din=DIM_IN, dh=64, da=DIM_ACT):
        super().__init__()
        self.h = dh
        self.embed = nn.Sequential(nn.Linear(din, 64), nn.ReLU())
        self.gru = nn.GRU(64, dh, batch_first=True)
        self.policy_head = nn.Sequential(nn.Linear(dh, 32), nn.ReLU(), nn.Linear(32, da))
        self.value_head = nn.Sequential(nn.Linear(dh, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, h):
        z = self.embed(x)
        y, h = self.gru(z, h)
        return self.policy_head(y), self.value_head(y), h

    def h0(self, b=1):
        return torch.zeros(1, b, self.h)


_net: Optional[Net] = None
_mem: Optional[torch.Tensor] = None
_t: int = 0


def _boot():
    global _net
    if _net is not None:
        return

    base = os.path.dirname(__file__)
    path = os.path.join(base, "ppo_be_ppo.pth")

    if not os.path.isfile(path):
        raise RuntimeError("missing weights")

    m = Net()
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    m.eval()
    _net = m


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _mem, _t
    _boot()

    if _mem is None or _t == 0:
        _mem = _net.h0(1)

    x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, 1, -1)
    logits, _, _mem = _net(x, _mem)

    if rng.random() < 0.2:
        idx = int(rng.integers(0, DIM_ACT))
    else:
        idx = int(logits[:, -1].argmax().item())

    _t += 1
    return ACTS[idx]

def reset_agent():
    global _mem, _t
    _mem = None
    _t = 0