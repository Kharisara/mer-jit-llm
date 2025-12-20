# # mer/env_meld_jitai.py

# import os
# from typing import Dict, List, Any, Iterable, Optional

# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset

# # ------------------------------------------------------------
# # Helper: safe .npy/.pt state loader
# # ------------------------------------------------------------
# def load_state_np(state_path: str, expected_dim: int = 512, device: str = "cpu") -> torch.Tensor:
#     """
#     Safely load a state stored as a .npy (preferred) or fall back to torch.load if needed.
#     Returns a torch.float32 tensor on the requested device with shape (expected_dim,).
#     Behavior:
#       - If state_path is missing/empty -> returns zeros.
#       - If file exists and is .npy -> np.load -> pad/trim to expected_dim.
#       - If file exists and not .npy -> try np.load(allow_pickle=True) then torch.load.
#       - Ensures finite values (replace non-finite with 0).
#     """
#     # defensive empty path
#     if not isinstance(state_path, str) or len(state_path.strip()) == 0:
#         return torch.zeros(expected_dim, dtype=torch.float32, device=device)

#     # try preferred numpy load first
#     try:
#         if state_path.endswith(".npy") or state_path.endswith(".npz"):
#             arr = np.load(state_path)
#         else:
#             # try np.load with allow_pickle (sometimes saved oddly)
#             try:
#                 arr = np.load(state_path, allow_pickle=True)
#             except Exception:
#                 # fallback to torch.load
#                 try:
#                     obj = torch.load(state_path, map_location="cpu")
#                     if isinstance(obj, np.ndarray):
#                         arr = obj
#                     elif hasattr(obj, "numpy"):
#                         arr = obj.numpy()
#                     elif isinstance(obj, dict):
#                         # common keys
#                         found = False
#                         for k in ("state", "embedding", "features", "z"):
#                             if k in obj:
#                                 val = obj[k]
#                                 if hasattr(val, "numpy"):
#                                     arr = val.numpy()
#                                 elif isinstance(val, np.ndarray):
#                                     arr = val
#                                 else:
#                                     arr = np.asarray(val)
#                                 found = True
#                                 break
#                         if not found:
#                             # try converting whole dict
#                             arr = np.asarray(obj)
#                     else:
#                         arr = np.asarray(obj)
#                 except Exception:
#                     # cannot load
#                     return torch.zeros(expected_dim, dtype=torch.float32, device=device)
#         arr = np.asarray(arr).ravel().astype(np.float32)
#     except Exception:
#         # any failure -> zeros
#         return torch.zeros(expected_dim, dtype=torch.float32, device=device)

#     # sanitize NaN / inf
#     if not np.isfinite(arr).all():
#         arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

#     # pad/trim to expected_dim
#     if arr.size < expected_dim:
#         padded = np.zeros(expected_dim, dtype=np.float32)
#         padded[: arr.size] = arr
#         arr = padded
#     elif arr.size > expected_dim:
#         arr = arr[:expected_dim]

#     t = torch.tensor(arr, dtype=torch.float32, device=device)
#     return t


# # ------------------------------------------------------------
# # 1. Emotion → valence mapping for MELD (4 classes)
# # ------------------------------------------------------------
# LABEL2VALENCE = {
#     "angry": -1.0,
#     "sad": -1.0,
#     "neutral": 0.0,
#     "happy": 1.0,
# }


# def label_to_valence(label: str) -> float:
#     return LABEL2VALENCE.get(label, 0.0)


# # ------------------------------------------------------------
# # 2. Offline JITAI / contextual bandit dataset
# # ------------------------------------------------------------
# class MELDJITAITransitionDataset(Dataset):
#     """
#     Offline JITAI / bandit dataset built from:
#       - CSV with TAV-context state paths
#       - Per-utterance emotion labels
#       - Dialogue structure (Dialogue_ID, Utterance_ID)

#     Each item is one transition between utterance t and t+1
#     in the SAME dialogue, where both have valid state vectors.

#     Returns a dict with:
#       state         : Tensor [state_dim]
#       action        : int (0 = no intervention, 1 = intervene)
#       reward        : float (valence_{t+1} - valence_t)
#       next_state    : Tensor [state_dim]
#       dialogue_id   : int
#       t_index       : int (utterance index within that dialogue)
#       label_t       : str
#       label_next    : str
#     """

#     def __init__(
#         self,
#         csv_path: str,
#         splits: Iterable[str] = ("train",),
#         device: str = "cpu",
#     ):
#         """
#         Args:
#             csv_path: path to meld_text_audio_video_arcface_states.csv
#             splits: which splits to use, e.g. ("train", "dev"), ("train","dev","test")
#             device: where to put the loaded state tensors ("cpu" is recommended)
#         """
#         super().__init__()
#         self.csv_path = csv_path
#         self.splits = tuple(splits)
#         self.device = torch.device(device)

#         df = pd.read_csv(csv_path)

#         # Keep only desired splits
#         if "split" in df.columns:
#             df = df[df["split"].isin(self.splits)].reset_index(drop=True)
#         else:
#             # no split column -> assume all rows are acceptable
#             df = df.reset_index(drop=True)

#         # Require state_path column
#         if "state_path" not in df.columns:
#             raise ValueError(
#                 "CSV must contain a 'state_path' column. "
#                 "Run extract_tav_context_states.py first."
#             )

#         # Filter rows that actually have a state_path (not NaN)
#         df = df[df["state_path"].notna()].reset_index(drop=True)

#         # Sanity: require dialogue structure
#         for col in ["Dialogue_ID", "Utterance_ID", "label"]:
#             if col not in df.columns:
#                 raise ValueError(f"CSV must have '{col}' column.")

#         self.df = df

#         # Build list of transitions (indices into df)
#         self.transitions: List[Dict[str, Any]] = []
#         self._build_transitions()

#         # Infer state dimension by loading one state via helper
#         if len(self.transitions) > 0:
#             first_state_path = self.transitions[0]["state_path_t"]
#             sample_state_t = load_state_np(first_state_path, expected_dim=512, device="cpu")
#             sample_state = sample_state_t.cpu().numpy()
#             if not isinstance(sample_state, np.ndarray):
#                 raise TypeError(
#                     f"State file at {first_state_path} is not a numpy array (after loading)."
#                 )
#             self.state_dim = int(sample_state.size)
#         else:
#             self.state_dim = 0

#         print(
#             f"[MELDJITAITransitionDataset] Splits={self.splits} "
#             f"Transitions: {len(self.transitions)}  State dim: {self.state_dim}"
#         )

#     # --------------------------------------------------------
#     # Build transitions by iterating over dialogues
#     # --------------------------------------------------------
#     def _build_transitions(self):
#         df = self.df

#         # group by Dialogue_ID
#         for d_id, df_d in df.groupby("Dialogue_ID"):
#             # sort by Utterance_ID inside this dialogue
#             df_d = df_d.sort_values("Utterance_ID")

#             idxs = df_d.index.to_list()
#             # create consecutive pairs (t, t+1)
#             for i in range(len(idxs) - 1):
#                 idx_t = idxs[i]
#                 idx_next = idxs[i + 1]

#                 row_t = df.loc[idx_t]
#                 row_next = df.loc[idx_next]

#                 # paths to state vectors
#                 s_path_t = row_t["state_path"]
#                 s_path_next = row_next["state_path"]

#                 # skip if any missing
#                 if not isinstance(s_path_t, str) or not isinstance(s_path_next, str):
#                     continue

#                 self.transitions.append(
#                     {
#                         "idx_t": idx_t,
#                         "idx_next": idx_next,
#                         "dialogue_id": int(row_t["Dialogue_ID"]),
#                         "utt_id_t": int(row_t["Utterance_ID"]),
#                         "utt_id_next": int(row_next["Utterance_ID"]),
#                         "label_t": str(row_t["label"]),
#                         "label_next": str(row_next["label"]),
#                         "state_path_t": s_path_t,
#                         "state_path_next": s_path_next,
#                     }
#                 )

#     def __len__(self):
#         return len(self.transitions)

#     # --------------------------------------------------------
#     # Synthetic behaviour policy for logged actions
#     # --------------------------------------------------------
#     @staticmethod
#     def _logging_policy(val_t: float) -> int:
#         """
#         Very simple rule-based behaviour policy:

#           if valence <= 0 --> action = 1 (intervene)
#           if valence >  0 --> action = 0 (no intervention)
#         """
#         return 1 if val_t <= 0.0 else 0

#     # --------------------------------------------------------
#     # Loading states + returning (s, a, r, s')
#     # --------------------------------------------------------
#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         tr = self.transitions[idx]

#         label_t = tr["label_t"]
#         label_next = tr["label_next"]

#         val_t = label_to_valence(label_t)
#         val_next = label_to_valence(label_next)

#         # Reward: change in valence between t and t+1
#         reward = float(val_next - val_t)

#         # Synthetic logged action
#         action = self._logging_policy(val_t)

#         # Load state and next_state using helper (ensures device placement and shape)
#         state = load_state_np(tr["state_path_t"], expected_dim=512, device=str(self.device))
#         next_state = load_state_np(tr["state_path_next"], expected_dim=512, device=str(self.device))

#         # Ensure tensors are 1D and float
#         state = state.view(-1).float()
#         next_state = next_state.view(-1).float()

#         return {
#             "state": state,  # [state_dim]
#             "action": torch.tensor(action, dtype=torch.long),
#             "reward": torch.tensor(reward, dtype=torch.float32),
#             "next_state": next_state,  # [state_dim]
#             "dialogue_id": torch.tensor(tr["dialogue_id"], dtype=torch.long),
#             "t_index": torch.tensor(tr["utt_id_t"], dtype=torch.long),
#             "label_t": label_t,
#             "label_next": label_next,
#         }


# # ------------------------------------------------------------
# # 3. Small usage demo (python -m mer.env_meld_jitai)
# # ------------------------------------------------------------
# if __name__ == "__main__":
#     this_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(this_dir)

#     csv_path = os.path.join(
#         project_root, "data", "processed", "meld_text_audio_video_arcface_states.csv"
#     )

#     # Use ALL splits just to see how many transitions we have overall
#     ds_all = MELDJITAITransitionDataset(
#         csv_path=csv_path,
#         splits=("train", "dev", "test"),
#         device="cpu",
#     )

#     print("Num transitions (all splits):", len(ds_all))
#     if len(ds_all) > 0:
#         sample = ds_all[0]
#         print("Sample keys:", sample.keys())
#         print("state shape:", sample["state"].shape)
#         print("action:", sample["action"].item())
#         print("reward:", sample["reward"].item())
#         print("label_t, label_next:", sample["label_t"], "→", sample["label_next"])

# mer/env_meld_jitai.py

# mer/env_meld_jitai.py

import os
from typing import Dict, List, Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ------------------------------------------------------------
# Helper: safe .npy/.pt state loader
# ------------------------------------------------------------
def load_state_np(state_path: str, expected_dim: int = 512, device: str = "cpu") -> torch.Tensor:
    if not isinstance(state_path, str) or len(state_path.strip()) == 0:
        return torch.zeros(expected_dim, dtype=torch.float32, device=device)

    try:
        if state_path.endswith(".npy") or state_path.endswith(".npz"):
            arr = np.load(state_path)
        else:
            try:
                arr = np.load(state_path, allow_pickle=True)
            except Exception:
                obj = torch.load(state_path, map_location="cpu")
                if isinstance(obj, np.ndarray):
                    arr = obj
                elif hasattr(obj, "numpy"):
                    arr = obj.numpy()
                elif isinstance(obj, dict):
                    for k in ("state", "embedding", "features", "z"):
                        if k in obj:
                            v = obj[k]
                            arr = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
                            break
                    else:
                        arr = np.asarray(obj)
                else:
                    arr = np.asarray(obj)
        arr = np.asarray(arr).ravel().astype(np.float32)
    except Exception:
        return torch.zeros(expected_dim, dtype=torch.float32, device=device)

    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.size < expected_dim:
        padded = np.zeros(expected_dim, dtype=np.float32)
        padded[:arr.size] = arr
        arr = padded
    elif arr.size > expected_dim:
        arr = arr[:expected_dim]

    return torch.tensor(arr, dtype=torch.float32, device=device)


# ------------------------------------------------------------
# Emotion → valence mapping (MELD)
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
# Offline JITAI dataset
# ------------------------------------------------------------
class MELDJITAITransitionDataset(Dataset):
    """
    Offline JITAI / contextual bandit dataset.

    reward_mode:
      - "original": r = valence_{t+1} - valence_t
      - "shaped":   small penalty for unnecessary intervention
    """

    def __init__(
        self,
        csv_path: str,
        splits: Iterable[str] = ("train",),
        device: str = "cpu",
        reward_mode: str = "original",
    ):
        super().__init__()
        self.csv_path = csv_path
        self.splits = tuple(splits)
        self.device = torch.device(device)
        self.reward_mode = reward_mode

        df = pd.read_csv(csv_path)

        if "split" in df.columns:
            df = df[df["split"].isin(self.splits)].reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        if "state_path" not in df.columns:
            raise ValueError("CSV must contain 'state_path' column.")

        df = df[df["state_path"].notna()].reset_index(drop=True)

        for col in ["Dialogue_ID", "Utterance_ID", "label"]:
            if col not in df.columns:
                raise ValueError(f"CSV must contain '{col}' column.")

        self.df = df
        self.transitions: List[Dict[str, Any]] = []
        self._build_transitions()

        if len(self.transitions) > 0:
            sample = load_state_np(self.transitions[0]["state_path_t"], device="cpu")
            self.state_dim = int(sample.numel())
        else:
            self.state_dim = 0

        print(
            f"[MELDJITAITransitionDataset] "
            f"Splits={self.splits}  "
            f"Transitions={len(self.transitions)}  "
            f"State dim={self.state_dim}  "
            f"Reward mode={self.reward_mode}"
        )

    # --------------------------------------------------------
    def _build_transitions(self):
        for _, df_d in self.df.groupby("Dialogue_ID"):
            df_d = df_d.sort_values("Utterance_ID")
            idxs = df_d.index.tolist()
            for i in range(len(idxs) - 1):
                r_t = self.df.loc[idxs[i]]
                r_n = self.df.loc[idxs[i + 1]]

                if not isinstance(r_t["state_path"], str) or not isinstance(r_n["state_path"], str):
                    continue

                self.transitions.append({
                    "dialogue_id": int(r_t["Dialogue_ID"]),
                    "utt_id_t": int(r_t["Utterance_ID"]),
                    "label_t": str(r_t["label"]),
                    "label_next": str(r_n["label"]),
                    "state_path_t": r_t["state_path"],
                    "state_path_next": r_n["state_path"],
                })

    # --------------------------------------------------------
    @staticmethod
    def _logging_policy(val_t: float) -> int:
        return 1 if val_t <= 0.0 else 0

    # --------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tr = self.transitions[idx]

        val_t = label_to_valence(tr["label_t"])
        val_next = label_to_valence(tr["label_next"])

        reward = float(val_next - val_t)
        action = self._logging_policy(val_t)

        # ---- Synthetic shaping (OPTIONAL) ----
        if self.reward_mode == "shaped":
            if val_t >= 0.0 and action == 1:
                reward -= 0.05

        state = load_state_np(tr["state_path_t"], device=str(self.device))
        next_state = load_state_np(tr["state_path_next"], device=str(self.device))

        return {
            "state": state.view(-1),
            "action": torch.tensor(action, dtype=torch.long),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_state": next_state.view(-1),
            "dialogue_id": torch.tensor(tr["dialogue_id"], dtype=torch.long),
            "t_index": torch.tensor(tr["utt_id_t"], dtype=torch.long),
            "label_t": tr["label_t"],
            "label_next": tr["label_next"],
        }

    def __len__(self):
        return len(self.transitions)


# ------------------------------------------------------------
# Demo
# ------------------------------------------------------------
if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)

    csv_path = os.path.join(
        project_root, "data", "processed", "meld_text_audio_video_arcface_states.csv"
    )

    ds = MELDJITAITransitionDataset(
    csv_path=args.csv,
    splits=("train",),
    device=args.device,
    reward_mode=args.reward_mode,  # <-- THIS LINE WAS MISSING OR WRONG
)


    print("Transitions:", len(ds))
    if len(ds) > 0:
        s = ds[0]
        print("Sample reward:", s["reward"].item())
