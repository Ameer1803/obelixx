import torch
import cv2
import numpy as np
import sys
import os
import argparse
from typing import Optional

# Add parent directory to path to import obelix
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from obelix import OBELIX
import torch.nn as nn


class RecurrentQNet(nn.Module):
    """Recurrent Q-Network for POMDP with LSTM."""

    def __init__(self, obs_dim=18, action_dim=5, hidden_dim=256, lstm_dim=256):
        super().__init__()

        # Encoder: MLP to process observations
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # LSTM: Recurrent memory
        self.lstm = nn.LSTM(hidden_dim, lstm_dim, batch_first=True)

        # Q-Head: Output Q-values per action
        self.head = nn.Linear(lstm_dim, action_dim)

    def forward(self, x, hidden=None):
        """
        Forward pass.

        Args:
            x: [batch, seq_len, obs_dim]
            hidden: (h, c) tuple or None

        Returns:
            q: [batch, seq_len, action_dim]
            hidden: (h, c) tuple for next step
        """
        # Encode observations
        z = self.enc(x)  # [batch, seq_len, hidden_dim]

        # Process through LSTM
        z, hidden = self.lstm(z, hidden)  # [batch, seq_len, lstm_dim]

        # Compute Q-values
        q = self.head(z)  # [batch, seq_len, action_dim]

        return q, hidden


def visualize_agent(model_path, num_episodes=5, scaling_factor=5, arena_size=500,
                    max_steps=1000, wall_obstacles=True, difficulty=0, box_speed=2,
                    render=True, seed=42):
    """
    Visualize the agent following the learned policy.

    Args:
        model_path: Path to the trained model weights
        num_episodes: Number of episodes to run
        scaling_factor: Environment scaling factor
        arena_size: Arena size in pixels
        max_steps: Maximum steps per episode
        wall_obstacles: Whether to include wall obstacles
        difficulty: Difficulty level (0=static, 2=blinking, 3=moving)
        box_speed: Speed of moving box
        render: Whether to render the environment
        seed: Random seed
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Initialize Environment
    print(f"\n[ENV] Initializing OBELIX...")
    print(f"  - Scaling factor: {scaling_factor}")
    print(f"  - Arena size: {arena_size}x{arena_size}")
    print(f"  - Max steps: {max_steps}")
    print(f"  - Wall obstacles: {wall_obstacles}")
    print(f"  - Difficulty: {difficulty}")
    print(f"  - Box speed: {box_speed}\n")

    env = OBELIX(
        scaling_factor=scaling_factor,
        arena_size=arena_size,
        max_steps=max_steps,
        wall_obstacles=wall_obstacles,
        difficulty=difficulty,
        box_speed=box_speed,
        seed=seed
    )

    # Initialize Model
    print(f"[MODEL] Loading from {model_path}...")
    model = RecurrentQNet(obs_dim=18, action_dim=5, hidden_dim=256, lstm_dim=256)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Loaded: {total_params:,} parameters")
    print(f"[MODEL] Architecture:")
    print(f"  - Encoder: 18 → 256 → 256")
    print(f"  - LSTM: 256-dim")
    print(f"  - Q-Head: 256 → 5\n")

    # Action mapping
    actions = ["L45", "L22", "FW", "R22", "R45"]
    action_names = {0: "L45", 1: "L22", 2: "FW", 3: "R22", 4: "R45"}

    print("="*80)
    print("STARTING POLICY VISUALIZATION")
    print("="*80 + "\n")

    total_rewards = []
    episode_lengths = []
    successes = 0
    action_counts = {a: 0 for a in actions}

    for episode in range(num_episodes):
        obs = env.reset(seed=seed + episode if seed is not None else None)
        done = False
        episode_reward = 0.0
        step = 0

        # Initialize hidden state for LSTM [batch=1, lstm_dim=256]
        hidden = (
            torch.zeros(1, 1, 256, device=device),
            torch.zeros(1, 1, 256, device=device)
        )

        print(f"\n{'─'*80}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'─'*80}")

        while not done:
            # Render the environment
            if render:
                env.render_frame()
                cv2.waitKey(30)  # 30ms delay for visualization

            # Prepare observation tensor [batch=1, seq_len=1, obs_dim=18]
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)

            # Get action from model (greedy policy)
            with torch.no_grad():
                q_values, hidden = model(obs_tensor, hidden)  # q_values: [1, 1, 5]
                q_vals_step = q_values[0, 0]  # [5]
                action_idx = q_vals_step.argmax().item()

            action = actions[action_idx]
            q_val_max = q_vals_step[action_idx].item()
            action_counts[action] += 1

            # Execute action
            obs, reward, done = env.step(action, render=False)
            episode_reward += reward
            step += 1

            # Print step info
            print(f"Step {step:3d} | Action: {action:>3s} | Q-val: {q_val_max:7.3f} | "
                  f"Reward: {reward:7.2f} | Total: {episode_reward:8.2f}", end='\r')

            if step >= max_steps:
                done = True

        total_rewards.append(episode_reward)
        episode_lengths.append(step)

        # Check success condition (bot attached and box touches boundary)
        if hasattr(env, 'enable_push') and env.enable_push:
            successes += 1

        print(f"\n✓ Episode {episode + 1} Complete")
        print(f"  - Total Reward: {episode_reward:.2f}")
        print(f"  - Episode Length: {step}")
        print(f"  - Box Attached: {env.enable_push if hasattr(env, 'enable_push') else 'N/A'}")

    # Print statistics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nReturn Statistics:")
    print(f"  - Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  - Max Reward:     {np.max(total_rewards):.2f}")
    print(f"  - Min Reward:     {np.min(total_rewards):.2f}")

    print(f"\nEpisode Statistics:")
    print(f"  - Avg Length: {np.mean(episode_lengths):.1f} steps")
    print(f"  - Success Rate: {(successes / num_episodes * 100):.1f}% ({successes}/{num_episodes})")

    print(f"\nAction Distribution:")
    total_actions = sum(action_counts.values())
    for action, count in action_counts.items():
        pct = (count / total_actions * 100) if total_actions > 0 else 0
        print(f"  - {action}: {count:4d} ({pct:5.1f}%)")

    print("="*80 + "\n")

    if render:
        cv2.destroyAllWindows()

    return {
        'rewards': total_rewards,
        'lengths': episode_lengths,
        'successes': successes,
        'action_counts': action_counts
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DDQRN agent policy on OBELIX")

    parser.add_argument(
        "-m", "--model",
        help="Path to trained model weights",
        type=str,
        default="ddqrn_obelix_weights.pt"
    )
    parser.add_argument(
        "-e", "--episodes",
        help="Number of episodes to visualize",
        type=int,
        default=5
    )
    parser.add_argument(
        "-sf", "--scaling_factor",
        help="Environment scaling factor",
        type=int,
        default=5
    )
    parser.add_argument(
        "--arena_size",
        help="Arena size in pixels",
        type=int,
        default=500
    )
    parser.add_argument(
        "--max_steps",
        help="Maximum steps per episode",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--wall_obstacles",
        help="Add wall obstacles",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--difficulty",
        help="Difficulty level (0=static box, 2=blinking, 3=moving+blinking)",
        type=int,
        default=0
    )
    parser.add_argument(
        "--box_speed",
        help="Speed of moving box",
        type=int,
        default=2
    )
    parser.add_argument(
        "--no-render",
        help="Disable environment rendering",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "-s", "--seed",
        help="Random seed",
        type=int,
        default=123
    )

    args = parser.parse_args()

    results = visualize_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        render=not args.no_render,
        seed=args.seed
    )