"""
Train a small JITAI policy on the offline transitions using a 1-step policy-gradient.

Usage:
python mer/train_jitai_policy.py \
  --csv data/processed/meld_text_audio_video_arcface_states.csv \
  --out_model models/jitai_policy.pt \
  --epochs 10 --batch_size 64 --lr 1e-4 --device cpu

Notes:
- This is a simple starting point. It treats each logged transition as a 1-step
  episode with reward = valence_{t+1} - valence_t (already in dataset).
- Because the dataset used a synthetic logging policy, use these results for
  debugging and policy initialization. For production you should incorporate
  offline-offpolicy evaluation (IPS/DR) and safety checks.
"""
import os
import argparse
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mer.env_meld_jitai import MELDJITAITransitionDataset

class MLPPolicy(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits

    def act_probs(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

def collate_batch(batch):
    # batch: list of dicts from dataset.__getitem__
    states = torch.stack([b["state"].float() for b in batch], dim=0)
    actions = torch.tensor([int(b["action"].item()) if hasattr(b["action"], "item") else int(b["action"]) for b in batch], dtype=torch.long)
    rewards = torch.tensor([float(b["reward"].item()) if hasattr(b["reward"], "item") else float(b["reward"]) for b in batch], dtype=torch.float32)
    return states, actions, rewards

def train(args):
    ds = MELDJITAITransitionDataset(csv_path=args.csv, splits=("train",), device=args.device)
    print("Dataset transitions:", len(ds), "state_dim:", ds.state_dim)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, drop_last=False, num_workers=0)

    device = torch.device(args.device)
    policy = MLPPolicy(state_dim=ds.state_dim, hidden=args.hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    # moving baseline for reward
    baseline = 0.0
    baseline_deque = deque(maxlen=1000)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_samples = 0
        total_reward = 0.0
        action_counts = torch.zeros(2, dtype=torch.long)

        policy.train()
        for states, actions, rewards in loader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)

            probs = policy.act_probs(states)  # [B, 2]
            dist = torch.distributions.Categorical(probs=probs)

            # compute baseline (moving average)
            batch_mean_reward = rewards.mean().item()
            baseline_deque.append(batch_mean_reward)
            baseline = float(sum(baseline_deque) / len(baseline_deque))

            # REINFORCE loss: -log pi(a|s) * (R - b)
            logp = dist.log_prob(actions)  # [B]
            adv = (rewards - baseline)
            loss = - (logp * adv).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item()) * states.size(0)
            total_reward += float(rewards.sum().item())
            total_samples += states.size(0)
            # tally actions under current policy argmax (for reporting)
            with torch.no_grad():
                a_pred = probs.argmax(dim=-1)
                for a in a_pred.cpu().tolist():
                    action_counts[a] += 1

        avg_loss = total_loss / (total_samples + 1e-9)
        avg_reward = total_reward / (total_samples + 1e-9)
        print(f"Epoch {epoch}/{args.epochs}  avg_loss={avg_loss:.6f}  avg_reward={avg_reward:.6f}  baseline={baseline:.4f}")
        print("Epoch action counts (policy argmax):", action_counts.tolist())

        # simple checkpoint each epoch
        os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
        torch.save(policy.state_dict(), args.out_model)

    # final evaluation on dataset: compute policy's action distribution and expected reward under greedy policy
    policy.eval()
    all_actions = []
    all_rewards = []
    with torch.no_grad():
        for states, actions, rewards in DataLoader(ds, batch_size=512, collate_fn=collate_batch):
            states = states.to(device)
            probs = policy.act_probs(states)
            greedy = probs.argmax(dim=-1).cpu()
            all_actions.extend(greedy.tolist())
            all_rewards.extend(rewards.numpy().tolist())
    import numpy as np
    all_actions = np.array(all_actions)
    print("Final greedy action distribution:", {int(i): int((all_actions==i).sum()) for i in [0,1]})
    print("Saved policy ->", args.out_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/processed/meld_text_audio_video_arcface_states.csv")
    parser.add_argument("--out_model", type=str, default="models/jitai_policy.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")


    args = parser.parse_args()
    train(args)
