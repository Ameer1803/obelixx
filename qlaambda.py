"""Offline trainer: Q(lambda) with eligibility traces for OBELIX.

Watkins's Q(λ): resets traces on exploratory actions, keeping off-policy
correctness. Eligibility traces propagate the terminal +2000 reward
backwards through the episode much faster than standard TD learning.

Run:
  python train_qlambda.py --obelix_py ./obelix_find.py --out weights_ql.pkl --episodes 20000
"""

from __future__ import annotations
import argparse, pickle, random
from collections import defaultdict
from tqdm import tqdm
import numpy as np

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
probs   = [0.15,  0.20,  0.30,  0.20,  0.15]
r_N = 2000

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIXFind

def obs_to_key(obs: np.ndarray) -> tuple:
    return tuple(obs.astype(int))

def epsilon_greedy(q_vals: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.choice(len(ACTIONS), p=probs))
    return int(np.argmax(q_vals))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str,   required=True)
    ap.add_argument("--out",            type=str,   default="weights_ql.pkl")
    ap.add_argument("--episodes",       type=int,   default=20000)
    ap.add_argument("--max_steps",      type=int,   default=450)
    ap.add_argument("--difficulty",     type=int,   default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed",      type=int,   default=2)
    ap.add_argument("--scaling_factor", type=int,   default=5)
    ap.add_argument("--arena_size",     type=int,   default=500)

    ap.add_argument("--alpha",          type=float, default=0.1)
    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--lambda_",        type=float, default=0.8)
    ap.add_argument("--eps_start",      type=float, default=1.0)
    ap.add_argument("--eps_end",        type=float, default=0.05)
    ap.add_argument("--eps_decay",      type=float, default=0.9995)  # per episode
    ap.add_argument("--seed",           type=int,   default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)
    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )

    # Q table: state -> array of 5 Q-values, default 0
    Q: dict = defaultdict(lambda: np.zeros(len(ACTIONS)))
    # Eligibility traces: same structure, reset each episode
    E: dict = {}

    epsilon = args.eps_start
    _boundary_recovery = 0
    _recovery_action   = 0

    pbar = tqdm(range(args.episodes), desc="Training")
    for ep in pbar:
        obs = env.reset(seed=args.seed + ep)
        key = obs_to_key(obs)

        E.clear()  # reset traces every episode

        a = epsilon_greedy(Q[key], epsilon, rng)
        ep_ret = 0.0
        _boundary_recovery = 0

        for steps in range(args.max_steps):

            # boundary recovery override (same logic as your D3QN)
            if _boundary_recovery > 0:
                a = _recovery_action
                _boundary_recovery -= 1
            elif obs[17] == 1:
                _recovery_action   = 0 if rng.random() < 0.5 else 4
                _boundary_recovery = 3

            elif np.sum(obs[:17]) == 0 and rng.random() < epsilon:
                # systematic sweep instead of random walk
                phase = steps % 15
                if phase < 10:
                    a = 2   # FW
                else:
                    a = 1   # L22 (slow turn)
            # ────────────────────────────────────────────────────────────
            
            else:
                a = epsilon_greedy(Q[key], epsilon, rng)

            obs2, r, done = env.step(ACTIONS[a], render=False)

            # turn penalty (same as your D3QN)
            if ACTIONS[a] in ["L45", "R45"]:
                r -= 0.4
            elif ACTIONS[a] in ["L22", "R22"]:
                r -= 0.2
            
            r = r/r_N

            ep_ret += float(r)

            key2 = obs_to_key(obs2)
            a2   = epsilon_greedy(Q[key2], epsilon, rng)  # next action

            # greedy action at next state (for Watkins's Q(λ))
            a2_greedy = int(np.argmax(Q[key2]))
            is_greedy = (a2 == a2_greedy)

            # TD error
            delta = r + args.gamma * np.max(Q[key2]) - Q[key][a]

            if np.isnan(delta):
                E.clear()
                continue

            # update trace for current state-action
            if key not in E:
                E[key] = np.zeros(len(ACTIONS))
            E[key][a] += 1.0  # accumulating traces

            # update Q and decay traces for all visited state-actions
            for s, e in E.items():
                Q[s] += args.alpha * delta * e
                E[s]  = args.gamma * args.lambda_ * e

            # Watkins's Q(λ): reset traces on exploratory action
            if not is_greedy:
                E.clear()

            key = key2
            obs = obs2
            a   = a2

            if done:
                # terminal update — no next state
                delta_term = r - Q[key][a]  # gamma * 0 since done
                if key not in E:
                    E[key] = np.zeros(len(ACTIONS))
                E[key][a] += 1.0
                for s, e in E.items():
                    Q[s] += args.alpha * delta_term * e
                break

        # decay epsilon per episode
        epsilon = max(args.eps_end, epsilon * args.eps_decay)

        pbar.set_postfix(
            return_=round(ep_ret, 2),
            eps=round(epsilon, 4),
            states=len(Q),
        )

    # save Q table
    with open(args.out, "wb") as f:
        pickle.dump(dict(Q), f)
    print(f"Saved Q-table with {len(Q)} states to {args.out}")

if __name__ == "__main__":
    main()