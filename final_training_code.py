"""Offline trainer: Double DQN + replay buffer (CPU) for OBELIX.

Run locally to create weights.pth, then submit agent.py + weights.pth.

Example:
python train_dqn.py --obelix_py ./obelix.py --out weights.pth --episodes 2000 --difficulty 0 --wall_obstacles

                        ALGORITHM: DOUBLE DEEP Q-NETWORK (DDQN)


Double DQN is one of the most widely used and reliable improvements over the original Deep Q-Network (DQN).

Main problems it solves:
Vanilla DQN often overestimates true action values.
This happens because the same network is used twice:
1. to pick the best-looking action in the next state (max)
2. to evaluate how good that action actually is

When Q-values are noisy (which they almost always are early in training),
this double usage creates optimistic bias → the agent thinks some
actions are much better than they really are → leads to unstable learning.

Double DQN solution:
Split the responsibilities:
• Use the online / main Q-network  to SELECT which action looks best
• Use the target Q-network to EVALUATE (give the actual value)

So instead of:

    target = r + γ × max_a Q_target(s', a)

We do:

    target = r + γ × Q_target( s',   argmax_a Q_online(s', a)   )

This small change dramatically reduces overestimation and makes learning
much more stable — especially in environments with large action spaces
or noisy rewards.

For More Details please refer to https://arxiv.org/pdf/1509.06461 .


"""

from __future__ import annotations
import argparse, random
from collections import deque
from dataclasses import dataclass
from typing import Deque

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
probs = [0.15, 0.20, 0.30, 0.20, 0.15]
k = 8
r_N = 1

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

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class Replay:
    def __init__(self, cap: int = 100_000):
        self.buf: Deque[Transition] = deque(maxlen=cap)
    def add(self, t: Transition):
        self.buf.append(t)
    def sample(self, batch: int):
        idx = np.random.choice(len(self.buf), size=batch, replace=False)
        items = [self.buf[i] for i in idx]
        s = np.stack([it.s for it in items]).astype(np.float32)
        a = np.array([it.a for it in items], dtype=np.int64)
        r = np.array([it.r for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d
    def __len__(self): return len(self.buf)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIXFind

_boundary_recovery: int = 0
_recovery_action: int = 0
_sweep_step: int = 0
_post_recovery_fw: int = 5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=750)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=10)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--target_sync", type=int, default=2000)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", type=str, default=None, help="Path to existing weights")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    OBELIX = import_obelix(args.obelix_py)

    q = DQN().to(device)
    tgt = DQN().to(device)

    # ---- LOAD PRETRAINED WEIGHTS ----
    if args.load is not None:
        print(f"Loading weights from {args.load}")
        state_dict = torch.load(args.load, map_location=device)
        q.load_state_dict(state_dict)
        tgt.load_state_dict(state_dict)   # keep target in sync initially
    else:
        tgt.load_state_dict(q.state_dict())

    tgt.eval()  

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay(args.replay)
    steps = 0

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    def policy(obs: np.ndarray, rng: np.random.Generator, model: DQN, state_buffer: deque, eps: float) -> str:
        """Epsilon-greedy policy for training."""
        global _boundary_recovery, _recovery_action, _sweep_step, _post_recovery_fw

        if len(state_buffer) == 0:
            for _ in range(k):
                state_buffer.append(obs)
        state_buffer.append(obs)
        state = np.concatenate(state_buffer)

        # Post recovery — force FW to move away from wall
        if _post_recovery_fw > 0:
            _post_recovery_fw -= 1
            if _post_recovery_fw == 0:
                # Reset state buffer — fresh context after wall escape
                state_buffer.clear()
                for _ in range(k):
                    state_buffer.append(obs)
                return ACTIONS[2]

        # Stuck recovery
        if _boundary_recovery > 0:
            _boundary_recovery -= 1
            return ACTIONS[_recovery_action]

        if obs[17] == 1:
            _recovery_action = 0 if rng.random() < 0.5 else 4
            _boundary_recovery = 4
            _post_recovery_fw = 10  # After turning, force 10 FW steps
            return ACTIONS[_recovery_action]

        # Blind — structured sweep
        if np.sum(obs[:17]) == 0:
            _sweep_step += 1
            phase = _sweep_step % 25
            return ACTIONS[2] if phase < 22 else ACTIONS[1]

        # Sensors firing — epsilon-greedy decision
        device = next(model.parameters()).device
        x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = model(x).squeeze(0).detach().cpu().numpy()

        if rng.random() < eps:
            return rng.choice(ACTIONS)
        return ACTIONS[int(np.argmax(q))]

    env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed
        )
    
    pbar = tqdm(range(args.episodes), desc="Training")
    for ep in pbar:

        if ep % 400 == 0 and ep > 0:
            ckpt_path = args.out.replace(".pth", f"_ep{ep}.pth")
            torch.save(q.cpu().state_dict(), ckpt_path)
            q.to(device)
            print(f"\nCheckpoint saved: {ckpt_path}")
        
        obs = env.reset(seed=args.seed + ep)
        state_buffer = deque(maxlen=k)
        for _ in range(k):
            state_buffer.append(obs)
        s = np.concatenate(state_buffer)
        ep_ret = 0.0
        _boundary_recovery = 0
        _recovery_action = 0


        for _ in range(args.max_steps):
            if len(state_buffer) == 0:
                for _ in range(k):
                    state_buffer.append(obs)

            eps = eps_by_step(steps)
            action = policy(obs, rng, q, state_buffer, eps)
            obs2, r, done = env.step(action, render=False)

            # Add a turn penalty to the reward based on the action taken
            if action in ["L45", "R45"]:
                r -= 0.1
            elif action in ["L22", "R22"]:
                r -= 0.05

            ep_ret += float(r)

            state_buffer.append(obs2)
            s2 = np.concatenate(state_buffer)

            # Feed augmented state to the model
            replay.add(Transition(s=s, a=ACTIONS.index(action), r=float(r), s2=s2, done=bool(done)))
            s = s2
            obs = obs2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                for _ in range(3):
                    sb, ab, rb, s2b, db = replay.sample(args.batch)
                    sb_t = torch.tensor(sb, dtype=torch.float32, device=device)
                    ab_t = torch.tensor(ab, device=device)
                    rb_t = torch.tensor(rb, device=device)
                    s2b_t = torch.tensor(s2b, dtype=torch.float32, device=device)
                    db_t = torch.tensor(db, device=device)

                    with torch.no_grad():
                        next_q = q(s2b_t)
                        next_a = torch.argmax(next_q, dim=1)
                        next_q_tgt = tgt(s2b_t)
                        next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
                        y = rb_t + args.gamma * (1.0 - db_t) * next_val

                    pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                    loss = nn.functional.smooth_l1_loss(pred, y)

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                    opt.step()

                for tgt_param, q_param in zip(tgt.parameters(), q.parameters()):
                    tgt_param.data.copy_(
                        args.tau * q_param.data + (1 - args.tau) * tgt_param.data
                    )

            if done:
                break

        pbar.set_postfix(
            return_=round(ep_ret,2),
            eps=round(eps_by_step(steps),3),
            replay=len(replay)
        )

    torch.save(q.cpu().state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()