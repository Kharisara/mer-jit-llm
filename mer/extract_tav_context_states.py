"""
mer/extract_tav_context_states.py

Goal:
 - For each MELD utterance in the CSV, ensure there is a 'state_path' pointing
   to a saved 1D numpy array (512 floats).
 - Prefer using existing video-based contextual states if present.
 - If missing, compute a text+audio (TA) fallback state:
     - text: CLS embedding from transformers (distilbert)
     - audio: log-mel features (librosa) reduced by time-mean and linearly up/down-sampled
 - Save states as .npy files and update CSV.

Notes:
 - This script is defensive: if transformers or librosa are not available it will
   still run but produce zeros for the missing parts.
 - The produced state vectors are float32 and length 512.
"""

import os
import sys
import argparse
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Try to import heavy deps (transformers, librosa)
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

# ---------------------------
# Configuration / defaults
# ---------------------------
STATE_DIM = 512
TEXT_DIM = 256   # how many dims reserved for text
AUDIO_DIM = 256  # reserved for audio
DEFAULT_TEXT_MODEL = "distilbert-base-uncased"

# ---------------------------
# Utilities
# ---------------------------
def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def is_valid_path(x) -> bool:
    return isinstance(x, str) and len(x) > 0 and (not pd.isna(x))


def load_vector_if_exists(path: str) -> Optional[np.ndarray]:
    if not is_valid_path(path):
        return None
    if not os.path.exists(path):
        return None
    # try np.load for .npy, .npz; try torch.load for .pt (but prefer .npy)
    try:
        if path.endswith(".npy") or path.endswith(".npz"):
            return np.load(path)
        else:
            # attempt np.load generally (it will fail for pickled torch)
            try:
                return np.load(path, allow_pickle=True)
            except Exception:
                # last attempt: try to use torch to load tensors and convert to numpy
                try:
                    import torch
                    obj = torch.load(path, map_location="cpu")
                    if isinstance(obj, np.ndarray):
                        return obj
                    elif hasattr(obj, "numpy"):
                        return obj.numpy()
                    elif isinstance(obj, dict):
                        # try common keys
                        for k in ["state", "embedding", "features"]:
                            if k in obj:
                                val = obj[k]
                                if hasattr(val, "numpy"):
                                    return val.numpy()
                                elif isinstance(val, np.ndarray):
                                    return val
                    return None
                except Exception:
                    return None
    except Exception:
        return None


# ---------------------------
# Text encoder (CLS extractor)
# ---------------------------
class TextEncoderFallback:
    def __init__(self, model_name=DEFAULT_TEXT_MODEL, device="cpu"):
        self.available = False
        if not TRANSFORMERS_AVAILABLE:
            print("[WARN] transformers not available: text embeddings will be fallback zeros")
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.device = device
            self.model.to(self.device)
            self.available = True
        except Exception as e:
            print(f"[WARN] Failed to init text encoder ({e}). Using zeros for text.")
            self.available = False

    def encode(self, text: str) -> np.ndarray:
        """
        Return a 1D numpy vector (model hidden_size, usually 768).
        We'll later slice/resize to TEXT_DIM.
        """
        if not self.available or text is None:
            return np.zeros(768, dtype=np.float32)
        try:
            # import torch here to avoid failing when torch isn't present elsewhere
            import torch
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attn)
                # distilbert / bert: last_hidden_state[:,0] is CLS token
                cls = out.last_hidden_state[:, 0].cpu().numpy().squeeze(0)
            return cls.astype(np.float32)
        except Exception as e:
            print(f"[WARN] Text encode failed: {e}")
            return np.zeros(768, dtype=np.float32)


# ---------------------------
# Audio encoder (simple log-mel -> time-mean)
# ---------------------------
def audio_logmel_mean(audio_path: str, sample_rate=16000, n_mels=64, n_fft=400, hop_length=160, win_length=400, max_duration=3.0):
    if not LIBROSA_AVAILABLE:
        return np.zeros((n_mels,), dtype=np.float32)
    if not is_valid_path(audio_path) or not os.path.exists(audio_path):
        return np.zeros((n_mels,), dtype=np.float32)
    try:
        wav, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        max_samples = int(max_duration * sample_rate)
        if len(wav) < max_samples:
            wav = np.pad(wav, (0, max_samples - len(wav)))
        else:
            wav = wav[:max_samples]
        mel = librosa.feature.melspectrogram(y=wav, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels, power=2.0)
        logmel = librosa.power_to_db(mel + 1e-6)
        # reduce by time-mean -> shape (n_mels,)
        mm = np.mean(logmel, axis=1)
        return mm.astype(np.float32)
    except Exception as e:
        print(f"[WARN] audio processing failed for {audio_path}: {e}")
        return np.zeros((n_mels,), dtype=np.float32)


# ---------------------------
# Build state vector
# ---------------------------
def build_ta_fallback_state(text: Optional[str], audio_path: Optional[str], text_encoder: Optional[TextEncoderFallback]) -> np.ndarray:
    """
    Compose a 512-d fallback state from text (TEXT_DIM) + audio (AUDIO_DIM).
    Strategy:
      - text part: distilbert CLS (768) -> take first TEXT_DIM (or pad/trim)
      - audio part: logmel mean (n_mels=64) -> upsample/pad to AUDIO_DIM by repeating + linear scaling
    """
    # TEXT part
    if text_encoder is not None and text_encoder.available and isinstance(text, str) and len(text.strip()) > 0:
        try:
            text_vec = text_encoder.encode(text)  # shape (768,)
        except Exception:
            text_vec = np.zeros(768, dtype=np.float32)
    else:
        text_vec = np.zeros(768, dtype=np.float32)

    # reduce/expand text to TEXT_DIM
    if TEXT_DIM <= len(text_vec):
        text_part = text_vec[:TEXT_DIM]
    else:
        text_part = np.pad(text_vec, (0, TEXT_DIM - len(text_vec)), mode="constant")

    # AUDIO part
    audio_mel = audio_logmel_mean(audio_path)  # shape (n_mels,)
    # simplest expansion to AUDIO_DIM: repeat and trim/pad
    if AUDIO_DIM <= len(audio_mel):
        audio_part = audio_mel[:AUDIO_DIM]
    else:
        # repeat tiles
        repeats = int(np.ceil(AUDIO_DIM / len(audio_mel)))
        audio_rep = np.tile(audio_mel, repeats)[:AUDIO_DIM]
        audio_part = audio_rep

    # final state
    state = np.concatenate([text_part.astype(np.float32), audio_part.astype(np.float32)], axis=0)
    # final safety: ensure STATE_DIM length
    if state.shape[0] < STATE_DIM:
        state = np.pad(state, (0, STATE_DIM - state.shape[0]), mode="constant")
    elif state.shape[0] > STATE_DIM:
        state = state[:STATE_DIM]
    return state.astype(np.float32)


# ---------------------------
# Main extraction flow
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to processed CSV (meld_text_audio_video_arcface.csv)")
    parser.add_argument("--ckpt", default=None, help="(optional) path to a contextual model checkpoint (not required)")
    parser.add_argument("--state-root", default=None, help="Directory to save .npy state vectors")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing state files")
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)

    csv_path = args.csv or os.path.join(project_root, "data", "processed", "meld_text_audio_video_arcface_states.csv")
    state_root = args.state_root or os.path.join(project_root, "data", "processed", "MELD_state_embeddings_tav_context")
    safe_mkdir(state_root)

    print("Input CSV:", csv_path)
    print("State root:", state_root)
    print("Using transformers:", TRANSFORMERS_AVAILABLE, "librosa:", LIBROSA_AVAILABLE)

    df = pd.read_csv(csv_path)
    # Ensure columns exist
    for col in ["Dialogue_ID", "Utterance_ID", "text"]:
        if col not in df.columns:
            raise RuntimeError(f"CSV missing required column '{col}'")

    # init text encoder if available
    text_encoder = None
    if TRANSFORMERS_AVAILABLE:
        device = "cpu"
        try:
            text_encoder = TextEncoderFallback(model_name=DEFAULT_TEXT_MODEL, device=device)
        except Exception:
            text_encoder = None

    # Iterate dialogues and utterances
    updated_rows = 0
    total = len(df)
    pbar = tqdm(range(total), desc="Processing utterances", ncols=120)
    for idx in pbar:
        row = df.iloc[idx]
        # skip if existing state_path and not overwrite
        existing = row.get("state_path", None) if "state_path" in row.index else None
        if is_valid_path(existing) and os.path.exists(existing) and not args.overwrite:
            pbar.set_postfix_str(f"skip idx={idx} (exists)")
            continue

        # Try to use existing video embedding fields first (if any)
        video_candidate = None
        for cand in ["state_video_path", "video_embedding", "video_emb_path", "face_path", "arcface_path"]:
            if cand in df.columns and is_valid_path(row.get(cand)):
                video_candidate = row.get(cand)
                break

        saved_state = None
        if video_candidate:
            vec = load_vector_if_exists(video_candidate)
            if vec is not None:
                v = np.asarray(vec).ravel().astype(np.float32)
                if v.shape[0] == STATE_DIM:
                    saved_state = v
                else:
                    tmp = v
                    if tmp.shape[0] < STATE_DIM:
                        tmp = np.pad(tmp, (0, STATE_DIM - tmp.shape[0]), mode="constant")
                    else:
                        tmp = tmp[:STATE_DIM]
                    saved_state = tmp

        # If saved_state none, attempt to use any global 'video_embedding' column (value might be stringified)
        if saved_state is None and "video_embedding" in df.columns:
            ve = row.get("video_embedding")
            if is_valid_path(ve) and os.path.exists(ve):
                vec = load_vector_if_exists(ve)
                if vec is not None:
                    v = np.asarray(vec).ravel().astype(np.float32)
                    if v.shape[0] == STATE_DIM:
                        saved_state = v
                    else:
                        tmp = v
                        if tmp.shape[0] < STATE_DIM:
                            tmp = np.pad(tmp, (0, STATE_DIM - tmp.shape[0]), mode="constant")
                        else:
                            tmp = tmp[:STATE_DIM]
                        saved_state = tmp

        # If we have a saved_state from video, save it as .npy and update CSV
        if saved_state is not None:
            target_name = f"d{int(row['Dialogue_ID'])}_u{int(row['Utterance_ID'])}.npy"
            target_path = os.path.join(state_root, target_name)
            np.save(target_path, saved_state.astype(np.float32))
            df.at[idx, "state_path"] = target_path
            updated_rows += 1
            pbar.set_postfix_str(f"video->saved idx={idx}")
            continue

        # Otherwise build TA fallback
        text = row.get("text", None) if "text" in df.columns else None
        audio_path = None
        # check common audio columns
        for cand in ["audio_path", "wav_path", "audiofile"]:
            if cand in df.columns and is_valid_path(row.get(cand)):
                audio_path = row.get(cand)
                break

        state_vec = build_ta_fallback_state(text=text, audio_path=audio_path, text_encoder=text_encoder)
        # if completely zero (no modalities), skip
        if np.allclose(state_vec, 0.0):
            pbar.set_postfix_str(f"skip idx={idx} (no modalities)")
            continue

        # Save .npy
        target_name = f"d{int(row['Dialogue_ID'])}_u{int(row['Utterance_ID'])}.npy"
        target_path = os.path.join(state_root, target_name)
        try:
            np.save(target_path, state_vec.astype(np.float32))
            df.at[idx, "state_path"] = target_path
            updated_rows += 1
            pbar.set_postfix_str(f"ta->saved idx={idx}")
        except Exception as e:
            pbar.set_postfix_str(f"ERROR saving idx={idx}")
            print(f"[WARN] Failed to save state for idx={idx} -> {e}")

    pbar.close()

    # Save CSV (overwrite)
    out_csv = csv_path
    backup_csv = csv_path + ".bak"
    try:
        # keep a quick backup
        if not os.path.exists(backup_csv):
            df.to_csv(backup_csv, index=False)
    except Exception:
        pass

    df.to_csv(out_csv, index=False)
    print(f"\nDone. Updated rows: {updated_rows}")
    print(f"Saved CSV to: {out_csv}  (backup: {backup_csv})")
    print("Tip: run the coverage check to inspect state_path coverage.")
    print("Example: python -c \"import pandas as pd; df=pd.read_csv('...'); print(df['state_path'].notna().sum())\"")


if __name__ == "__main__":
    main()
