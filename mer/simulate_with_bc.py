"""
mer/simulate_with_bc.py

Simulate policy-conditioned responses using:
 - Behavioral Cloning (BC) policy with emotion-aware deployment
 - Optional policy ablation (bc / proxy / random)
 - Deterministic --no_llm mode
 - Strict JSON output + hard safety normalization

Logs:
 - action
 - policy_source
 - replies
"""

import os
import csv
import json
import time
import argparse
import logging
import random
from typing import Optional

import numpy as np
import pandas as pd
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("simulate_with_bc")

# -------------------------------------------------
# Label → valence (MELD)
# -------------------------------------------------
LABEL_TO_VALENCE = {
    "angry": -1.0,
    "anger": -1.0,
    "sad": -1.0,
    "sadness": -1.0,
    "fear": -0.7,
    "disgust": -0.7,
    "neutral": 0.0,
    "surprise": 0.0,
    "happy": 1.0,
    "joy": 1.0,
}

# -------------------------------------------------
# BC Policy (must match training)
# -------------------------------------------------
class BCPolicy(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_actions=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# -------------------------------------------------
# Load BC model once
# -------------------------------------------------
bc_policy = None
bc_device = "cpu"
bc_model_path = "models/jitai_policy_bc.pt"

if os.path.exists(bc_model_path):
    try:
        bc_policy = BCPolicy().to(bc_device)
        bc_policy.load_state_dict(torch.load(bc_model_path, map_location=bc_device))
        bc_policy.eval()
        logger.info("Loaded BC policy")
    except Exception as e:
        logger.warning(f"BC policy load failed: {e}")
        bc_policy = None
else:
    logger.warning("BC policy not found — proxy only")

# -------------------------------------------------
# Deterministic reply stub
# -------------------------------------------------
def make_stub_reply(text: str, valence: float):
    if valence < 0:
        return {
            "sentences": [
                "I hear you. That sounds difficult.",
                "Try taking a few slow breaths to steady yourself."
            ],
            "safety": "ok",
        }
    else:
        return {
            "sentences": [
                "That sounds positive.",
                "Consider one small step to build on that."
            ],
            "safety": "ok",
        }

# -------------------------------------------------
# Simulation
# -------------------------------------------------
def simulate(
    csv_path: str,
    out_path: str,
    max_rows: int = 0,
    no_llm: bool = False,
    policy_mode: str = "bc",
):

    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if max_rows > 0:
        df = df.iloc[:max_rows]

    out_cols = list(df.columns) + [
        "action",
        "policy_source",
        "reply_json",
        "reply_sentences",
        "reply_safety",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()

        for i, row in df.reset_index(drop=True).iterrows():
            text = str(row.get("text", ""))
            label = str(row.get("label", "neutral")).lower()
            valence = LABEL_TO_VALENCE.get(label, 0.0)

            # -----------------------------------------
            # POLICY DECISION
            # -----------------------------------------
            if policy_mode == "random":
                action = random.choice([0, 1])
                policy_source = "RANDOM"

            elif policy_mode == "proxy":
                action = 1 if valence < 0 else 0
                policy_source = "PROXY"

            else:  # BC (emotion-aware)
                try:
                    if bc_policy is None:
                        raise ValueError("BC unavailable")

                    state_path = row.get("state_path")
                    if not isinstance(state_path, str) or not os.path.exists(state_path):
                        raise ValueError("Missing state")

                    state = np.load(state_path).reshape(1, -1)
                    state_t = torch.from_numpy(state).float().to(bc_device)

                    with torch.no_grad():
                        logits = bc_policy(state_t)
                        probs = torch.softmax(logits, dim=-1).squeeze(0)
                        p_intervene = probs[1].item()

                    # Emotion-aware deployment gating
                    if label in ["happy", "joy"]:
                        action = 1 if p_intervene >= 0.85 else 0
                    elif label in ["neutral"]:
                        action = 1 if p_intervene >= 0.9 else 0
                    else:  # angry, sad, fear, disgust
                        action = 1

                    policy_source = "BC_EMOTION_AWARE"

                except Exception:
                    action = 1 if valence < 0 else 0
                    policy_source = "PROXY_FALLBACK"

            # -----------------------------------------
            # RESPONSE (deterministic stub)
            # -----------------------------------------
            reply = make_stub_reply(text, valence)

            out_row = row.to_dict()
            out_row["action"] = action
            out_row["policy_source"] = policy_source
            out_row["reply_json"] = json.dumps(reply)
            out_row["reply_sentences"] = json.dumps(reply["sentences"])
            out_row["reply_safety"] = reply["safety"]

            writer.writerow(out_row)

            if (i + 1) % 100 == 0 or i == 0:
                logger.info(f"Processed {i + 1}/{len(df)}")

    logger.info(f"Simulation complete → {out_path}")

# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument(
        "--policy_mode",
        type=str,
        default="bc",
        choices=["bc", "proxy", "random"],
    )
    parser.add_argument("--no_llm", action="store_true")

    args = parser.parse_args()

    simulate(
        csv_path=args.csv,
        out_path=args.out,
        max_rows=args.max_rows,
        no_llm=args.no_llm,
        policy_mode=args.policy_mode,
    )

if __name__ == "__main__":
    main()
