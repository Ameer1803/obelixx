from __future__ import annotations
import os, pickle
import numpy as np
from collections import defaultdict

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

_Q = None
_boundary_recovery = 0
_recovery_action   = 0

def _load_once():
    global _Q
    if _Q is not None:
        return
    here  = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights_ql.pkl")
    with open(wpath, "rb") as f:
        raw = pickle.load(f)
    # wrap in defaultdict so unseen states return zeros
    _Q = defaultdict(lambda: np.zeros(len(ACTIONS)), raw)

def obs_to_key(obs: np.ndarray) -> tuple:
    return tuple(obs.astype(int))

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _boundary_recovery, _recovery_action
    _load_once()

    # boundary recovery
    if _boundary_recovery > 0:
        _boundary_recovery -= 1
        return ACTIONS[_recovery_action]

    if obs[17] == 1:
        _recovery_action   = 0 if rng.random() < 0.5 else 4
        _boundary_recovery = 3
        return ACTIONS[_recovery_action]

    key  = obs_to_key(obs)
    best = int(np.argmax(_Q[key]))
    return ACTIONS[best]