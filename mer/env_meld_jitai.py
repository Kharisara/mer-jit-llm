# mer/env_meld_jitai.py

import os
from typing import Dict, List, Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ------------------------------------------------------------
# 1. Emotion → valence mapping for MELD (4 classes)
# ------------------------------------------------------------

LABEL2VALENCE = {
    "angry": -1.0,
    "sad": -1.0,
    "neutral": 0.0,
    "happy": 1.0,
}


def label_to_valence(label: str) -> float:
    return LABEL2VALENCE.get(label, 0.0)


# ------------------------------------------------------------
# 2. Offline JITAI / contextual bandit dataset
# ------------------------------------------------------------

class MELDJITAITransitionDataset(Dataset):
    """
    Offline JITAI / bandit dataset built from:
      - CSV with TAV-context state paths
      - Per-utterance emotion labels
      - Dialogue structure (Dialogue_ID, Utterance_ID)

    Each item is one transition between utterance t and t+1
    in the SAME dialogue, where both have valid state vectors.

    Returns a dict with:
      state         : Tensor [state_dim]
      action        : int (0 = no intervention, 1 = intervene)
      reward        : float (valence_{t+1} - valence_t)
      next_state    : Tensor [state_dim]
      dialogue_id   : int
      t_index       : int (utterance index within that dialogue)
      label_t       : str
      label_next    : str
    """

    def __init__(
        self,
        csv_path: str,
        splits: Iterable[str] = ("train",),
        device: str = "cpu",
    ):
        """
        Args:
            csv_path: path to meld_text_audio_video_arcface_states.csv
            splits: which splits to use, e.g. ("train", "dev"), ("train","dev","test")
            device: where to put the loaded state tensors ("cpu" is recommended)
        """
        super().__init__()
        self.csv_path = csv_path
        self.splits = tuple(splits)
        self.device = torch.device(device)

        df = pd.read_csv(csv_path)

        # Keep only desired splits
        df = df[df["split"].isin(self.splits)].reset_index(drop=True)

        # Require state_path column
        if "state_path" not in df.columns:
            raise ValueError(
                "CSV must contain a 'state_path' column. "
                "Run extract_tav_context_states.py first."
            )

        # Filter rows that actually have a state_path (not NaN)
        df = df[df["state_path"].notna()].reset_index(drop=True)

        # Sanity: require dialogue structure
        for col in ["Dialogue_ID", "Utterance_ID", "label"]:
            if col not in df.columns:
                raise ValueError(f"CSV must have '{col}' column.")

        self.df = df

        # Build list of transitions (indices into df)
        self.transitions: List[Dict[str, Any]] = []
        self._build_transitions()

        # Infer state dimension by loading one state via numpy
        if len(self.transitions) > 0:
            first_state_path = self.transitions[0]["state_path_t"]
            try:
                sample_state = np.load(first_state_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load sample state from {first_state_path}. "
                    f"Check that extract_tav_context_states.py saved .npy files. "
                    f"Error: {e}"
                )

            if not isinstance(sample_state, np.ndarray):
                raise TypeError(
                    f"State file at {first_state_path} is not a numpy array."
                )

            self.state_dim = int(sample_state.size)
        else:
            self.state_dim = 0

        print(
            f"[MELDJITAITransitionDataset] Splits={self.splits} "
            f"Transitions: {len(self.transitions)}  State dim: {self.state_dim}"
        )

    # --------------------------------------------------------
    # Build transitions by iterating over dialogues
    # --------------------------------------------------------
    def _build_transitions(self):
        df = self.df

        # group by Dialogue_ID
        for d_id, df_d in df.groupby("Dialogue_ID"):
            # sort by Utterance_ID inside this dialogue
            df_d = df_d.sort_values("Utterance_ID")

            idxs = df_d.index.to_list()
            # create consecutive pairs (t, t+1)
            for i in range(len(idxs) - 1):
                idx_t = idxs[i]
                idx_next = idxs[i + 1]

                row_t = df.loc[idx_t]
                row_next = df.loc[idx_next]

                # paths to state vectors
                s_path_t = row_t["state_path"]
                s_path_next = row_next["state_path"]

                # skip if any missing
                if not isinstance(s_path_t, str) or not isinstance(s_path_next, str):
                    continue

                self.transitions.append(
                    {
                        "idx_t": idx_t,
                        "idx_next": idx_next,
                        "dialogue_id": int(row_t["Dialogue_ID"]),
                        "utt_id_t": int(row_t["Utterance_ID"]),
                        "utt_id_next": int(row_next["Utterance_ID"]),
                        "label_t": str(row_t["label"]),
                        "label_next": str(row_next["label"]),
                        "state_path_t": s_path_t,
                        "state_path_next": s_path_next,
                    }
                )

    def __len__(self):
        return len(self.transitions)

    # --------------------------------------------------------
    # Synthetic behaviour policy for logged actions
    # --------------------------------------------------------
    @staticmethod
    def _logging_policy(val_t: float) -> int:
        """
        Very simple rule-based behaviour policy:

          if valence <= 0 --> action = 1 (intervene)
          if valence >  0 --> action = 0 (no intervention)
        """
        return 1 if val_t <= 0.0 else 0

    # --------------------------------------------------------
    # Loading states + returning (s, a, r, s')
    # --------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tr = self.transitions[idx]

        label_t = tr["label_t"]
        label_next = tr["label_next"]

        val_t = label_to_valence(label_t)
        val_next = label_to_valence(label_next)

        # Reward: change in valence between t and t+1
        reward = float(val_next - val_t)

        # Synthetic logged action
        action = self._logging_policy(val_t)

        # Load state and next_state from .npy
        try:
            state_np = np.load(tr["state_path_t"])
            next_state_np = np.load(tr["state_path_next"])
        except Exception as e:
            raise RuntimeError(
                f"Failed to load states for transition "
                f"(t: {tr['state_path_t']}, next: {tr['state_path_next']}): {e}"
            )

        if not isinstance(state_np, np.ndarray) or not isinstance(
            next_state_np, np.ndarray
        ):
            raise TypeError("Loaded state is not a numpy array.")

        state = torch.from_numpy(state_np).view(-1).to(self.device).float()
        next_state = torch.from_numpy(next_state_np).view(-1).to(self.device).float()

        return {
            "state": state,  # [state_dim]
            "action": torch.tensor(action, dtype=torch.long),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_state": next_state,  # [state_dim]
            "dialogue_id": torch.tensor(tr["dialogue_id"], dtype=torch.long),
            "t_index": torch.tensor(tr["utt_id_t"], dtype=torch.long),
            "label_t": label_t,
            "label_next": label_next,
        }


# ------------------------------------------------------------
# 3. Small usage demo (python -m mer.env_meld_jitai)
# ------------------------------------------------------------
if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)

    csv_path = os.path.join(
        project_root, "data", "processed", "meld_text_audio_video_arcface_states.csv"
    )

    # Use ALL splits just to see how many transitions we have overall
    ds_all = MELDJITAITransitionDataset(
        csv_path=csv_path,
        splits=("train", "dev", "test"),
        device="cpu",
    )

    print("Num transitions (all splits):", len(ds_all))
    if len(ds_all) > 0:
        sample = ds_all[0]
        print("Sample keys:", sample.keys())
        print("state shape:", sample["state"].shape)
        print("action:", sample["action"].item())
        print("reward:", sample["reward"].item())
        print("label_t, label_next:", sample["label_t"], "→", sample["label_next"])
