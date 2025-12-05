# mer/extract_tav_context_states.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset_multimodal_meld_tav_context import MELDDialogueTAVDataset
from .model_multimodal_tav_context import ContextualMultimodalTAVClassifier


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))   # .../mer
    project_root = os.path.dirname(this_dir)                # .../mer-jit-llm

    csv_path = os.path.join(
        project_root, "data", "processed", "meld_text_audio_video_arcface.csv"
    )
    ckpt_path = os.path.join(
        project_root, "checkpoints_multimodal_tav_context", "best_multimodal_tav_context.pt"
    )

    # where to save state vectors
    state_root = os.path.join(
        project_root, "data", "processed", "MELD_state_embeddings_tav_context"
    )
    os.makedirs(state_root, exist_ok=True)

    # output CSV with new column
    csv_out_path = os.path.join(
        project_root, "data", "processed", "meld_text_audio_video_arcface_states.csv"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Input CSV:", csv_path)
    print("Checkpoint:", ckpt_path)
    print("State root:", state_root)
    print("Output CSV:", csv_out_path)

    # -------------------------
    # Load label mapping & config from checkpoint
    # -------------------------
    ckpt = torch.load(ckpt_path, map_location=device)
    label2id = ckpt["label2id"]
    id2label = ckpt["id2label"]
    num_labels = len(label2id)

    # you can tweak these if needed
    text_model_name = "distilbert-base-uncased"
    max_length = 128
    n_mels = 64
    video_dim = 512

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # -------------------------
    # Build dialogue-level datasets (train+dev+test)
    # -------------------------
    df = pd.read_csv(csv_path)

    splits = ["train", "dev", "test"]
    all_state_paths = {}  # utterance_id -> state_path

    model = ContextualMultimodalTAVClassifier(
        num_labels=num_labels,
        text_model_name=text_model_name,
        text_proj_dim=256,
        audio_proj_dim=128,
        video_input_dim=video_dim,
        video_proj_dim=128,
        hidden_dim=256,
        bidirectional=True,
        class_weights=None,   # frozen, no training here
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        for split in splits:
            print(f"\n[Split: {split}]")
            ds = MELDDialogueTAVDataset(
                csv_path=csv_path,
                tokenizer=tokenizer,
                label2id=label2id,
                split=split,
                max_length=max_length,
                n_mels=n_mels,
                video_dim=video_dim,
            )
            loader = DataLoader(ds, batch_size=1, shuffle=False)

            for batch in tqdm(loader, desc=f"Encoding {split} dialogues"):
                input_ids = batch["input_ids"].to(device)          # [1, T, L]
                attention_mask = batch["attention_mask"].to(device)
                logmel = batch["logmel"].to(device)                # [1, T, 1, n_mels, Tspec]
                video_emb = batch["video_emb"].to(device)          # [1, T, video_dim]
                utterance_ids = batch["utterance_ids"][0]          # list of length T

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logmel=logmel,
                    video_emb=video_emb,
                    labels=None,
                    return_states=True,
                )
                states = out["states"].squeeze(0).cpu().numpy()    # [T, state_dim]

                # Save each utterance's state vector
                for uid, state_vec in zip(utterance_ids, states):
                    # uid expected like "diaXXX_uttY" or similar
                    fname = f"{uid}.npy"
                    path = os.path.join(state_root, fname)
                    np.save(path, state_vec.astype("float32"))
                    all_state_paths[uid] = path

    # -------------------------
    # Write updated CSV with state_path column
    # -------------------------
    if "state_path" not in df.columns:
        df["state_path"] = pd.NA

    num_updated = 0
    for i, row in df.iterrows():
        uid = row.get("utterance_id", None)
        if uid in all_state_paths:
            df.at[i, "state_path"] = all_state_paths[uid]
            num_updated += 1

    print(f"\nTotal utterances with state vectors: {len(all_state_paths)}")
    print(f"Rows updated in CSV: {num_updated}")

    df.to_csv(csv_out_path, index=False)
    print("Saved updated CSV with 'state_path' to:", csv_out_path)


if __name__ == "__main__":
    main()
