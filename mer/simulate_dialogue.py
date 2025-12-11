# mer/simulate_dialogue.py

import os
import argparse
from typing import Optional

import pandas as pd
import torch

from .env_meld_jitai import LABEL2VALENCE, label_to_valence
from .train_jitai_policy import JITAIPolicyNet


def load_policy(project_root: str, device: torch.device) -> JITAIPolicyNet:
    """
    Load the trained JITAI policy network from checkpoints_jitai/jitai_policy_best.pt
    """
    ckpt_path = os.path.join(project_root, "checkpoints_jitai", "jitai_policy_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Could not find JITAI checkpoint at {ckpt_path}. "
            "Make sure you ran train_jitai_policy.py first."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dim = ckpt["state_dim"]

    policy = JITAIPolicyNet(state_dim=state_dim)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.to(device)
    policy.eval()

    print(f"[simulate_dialogue] Loaded JITAI policy from {ckpt_path}")
    print(f"[simulate_dialogue] State dim = {state_dim}")
    return policy


def pick_dialogue(
    df: pd.DataFrame,
    split: str,
    dialogue_id: Optional[int] = None,
) -> int:
    """
    Choose a Dialogue_ID to simulate.
    If dialogue_id is None, pick the first dialogue in the given split
    that has at least one non-null state_path.
    """
    df_split = df[df["split"] == split]

    if dialogue_id is not None:
        if dialogue_id not in df_split["Dialogue_ID"].unique():
            raise ValueError(
                f"Dialogue_ID {dialogue_id} not found in split='{split}'."
            )
        return dialogue_id

    # auto-pick: first dialogue with at least one state_path
    for d_id, df_d in df_split.groupby("Dialogue_ID"):
        if df_d["state_path"].notna().any():
            print(f"[simulate_dialogue] Auto-selected Dialogue_ID={d_id} in split='{split}'")
            return int(d_id)

    raise RuntimeError(f"No dialogues with state_path found in split='{split}'.")


def build_llm_prompt(
    user_text: str,
    emotion_label: str,
    valence: float,
    action: int,
) -> str:
    """
    Construct a prompt you would send to an LLM.
    This DOES NOT actually call an LLM – it just returns the prompt string.
    """

    action_str = "offer a supportive, empathic response" if action == 1 else \
                 "respond briefly and neutrally without attempting an intervention"

    # You can tweak this template however you like for the thesis/demo.
    prompt = f"""
You are an empathic mental health support assistant.

The user's current utterance:
\"\"\"{user_text}\"\"\"

Detected emotion label: {emotion_label}
Mapped valence: {valence:+.1f}

JITAI policy action (0 = no intervention, 1 = supportive intervention): {action}

Your task: {action_str}.

Guidelines:
- Be concise (2–4 sentences).
- Be non-judgmental and validating.
- Do NOT give medical diagnoses.
- If you suggest actions, keep them small and practical.

Now write your reply to the user.
""".strip()

    return prompt


def simulate_dialogue(
    csv_path: str,
    split: str = "test",
    dialogue_id: Optional[int] = None,
    max_turns: Optional[int] = None,
    device: str = "cpu",
):
    device = torch.device(device)

    # --------------------------------------------------------
    # Load CSV with states
    # --------------------------------------------------------
    df = pd.read_csv(csv_path)

    if "state_path" not in df.columns:
        raise ValueError(
            "CSV does not contain 'state_path'. "
            "Run extract_tav_context_states.py first."
        )

    # --------------------------------------------------------
    # Resolve project_root and load policy
    # --------------------------------------------------------
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)
    policy = load_policy(project_root, device)

    # --------------------------------------------------------
    # Pick dialogue to simulate
    # --------------------------------------------------------
    d_id = pick_dialogue(df, split=split, dialogue_id=dialogue_id)

    df_d = df[(df["split"] == split) & (df["Dialogue_ID"] == d_id)].copy()
    df_d = df_d.sort_values("Utterance_ID").reset_index(drop=True)

    print(f"\n[simulate_dialogue] Simulating Dialogue_ID={d_id}, split='{split}'")
    print(f"Total utterances in dialogue: {len(df_d)}")
    print(f"Utterances with state_path: {df_d['state_path'].notna().sum()}")

    print("\n================= SIMULATED DIALOGUE =================\n")

    num_turns = 0
    for _, row in df_d.iterrows():
        speaker = row.get("speaker", "Unknown")
        text = str(row.get("text", ""))
        label = str(row.get("label", "neutral"))

        print(f"Utterance {int(row['Utterance_ID'])} | Speaker: {speaker}")
        print(f"USER: {text}")
        print(f"  (MELD label: {label})")

        if pd.isna(row["state_path"]):
            print("  [No state_path available → skipping JITAI / LLM for this turn]\n")
            continue

        # ----------------------------------------------------
        # Load z_t state vector
        # ----------------------------------------------------
        state_path = row["state_path"]
        try:
            z_t = torch.from_numpy(
                __import__("numpy").load(state_path)
            ).float().to(device)
        except Exception as e:
            print(f"  [WARN] Failed to load state from {state_path}: {e}")
            print("  [Skipping JITAI / LLM for this turn]\n")
            continue

        if z_t.ndim > 1:
            z_t = z_t.view(-1)

        # ----------------------------------------------------
        # Compute valence and JITAI action
        # ----------------------------------------------------
        val = label_to_valence(label)
        with torch.no_grad():
            logits = policy(z_t.unsqueeze(0))   # [1, 2]
            action = int(logits.argmax(dim=-1).item())

        action_name = "no intervention (0)" if action == 0 else "supportive intervention (1)"
        print(f"  → Valence: {val:+.1f}, JITAI policy action: {action_name}")

        # ----------------------------------------------------
        # Build LLM prompt (not actually calling an LLM here)
        # ----------------------------------------------------
        prompt = build_llm_prompt(
            user_text=text,
            emotion_label=label,
            valence=val,
            action=action,
        )

        print("\n  --- LLM PROMPT (for debugging / inspection) ---")
        print(prompt)
        print("  --- END PROMPT ---")

        # Dummy placeholder instead of real LLM call:
        print("ASSISTANT (dummy): [LLM-generated supportive reply would appear here.]\n")

        num_turns += 1
        if max_turns is not None and num_turns >= max_turns:
            print(f"[simulate_dialogue] Reached max_turns={max_turns}, stopping.")
            break

    print("\n================= END OF SIMULATION =================\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate a MELD dialogue with MER state + JITAI policy + LLM prompt."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Which split to sample dialogue from (default: test).",
    )
    parser.add_argument(
        "--dialogue-id",
        type=int,
        default=None,
        help="Specific Dialogue_ID to simulate (default: auto-select one with states).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Optional maximum number of utterances to simulate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for policy network (e.g., 'cpu' or 'cuda').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)

    csv_path = os.path.join(
        project_root,
        "data",
        "processed",
        "meld_text_audio_video_arcface_states.csv",
    )

    simulate_dialogue(
        csv_path=csv_path,
        split=args.split,
        dialogue_id=args.dialogue_id,
        max_turns=args.max_turns,
        device=args.device,
    )
