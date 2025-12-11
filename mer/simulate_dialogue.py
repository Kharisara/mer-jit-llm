"""
mer/simulate_dialogue.py

Loads metadata CSV (with state_path), projects states with pre-fit PCA (models/pca_states_2.pkl),
builds a compact prompt (valence, label, z_proj, short history), calls OpenAI (if OPENAI_API_KEY set)
or a safe stub, and writes simulation_outputs.csv with columns:

dialogue_id, utt_id, text, label, valence, action, z_x, z_y, reply

Usage (from project root):
python mer/simulate_dialogue.py --metadata_csv data/processed/meld_text_audio_video_arcface_states.csv --pca_path models/pca_states_2.pkl --out simulation_outputs.csv
"""
import os
import csv
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib

def load_pca(pca_path: str):
    if not os.path.exists(pca_path):
        raise FileNotFoundError(pca_path)
    return joblib.load(pca_path)

def build_small_prompt(text: str, label: str, valence: float, zproj: np.ndarray, action: int, history: List[str]) -> str:
    h = ""
    if history:
        h = "\nConversation history (recent):\n" + "\n".join(history[-3:])
    prompt = f"""You are an empathic mental-health support assistant.
User utterance: "{text}"
Detected emotion: {label} (valence={valence})
Affective projection: x={zproj[0]:.3f}, y={zproj[1]:.3f}
Policy action: {action}  (0=no-action, 1=intervene)
{h}
Task: Provide a concise (2-4 sentence) supportive reply. No diagnoses. If action==1, validate and give one practical suggestion.
Safety: If the user expresses self-harm or imminent danger, include a short escalation phrase and do NOT give instructions for self-harm.
"""
    return prompt

def call_openai_chat(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7, api_key: Optional[str] = None) -> str:
    # If openai is available and API key set, call it; otherwise return a safe stub
    try:
        import openai
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":"You are an empathic assistant."},
                      {"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=150
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        # deterministic, safe stub reply
        return ("Thanks for sharing that — I’m sorry you’re going through this. "
                "If you can, try a short grounding exercise (5 deep breaths) and consider "
                "one small action right now, e.g., stepping outside for two minutes.")

def simulate(metadata_csv: str, pca_path: str, out_csv: str, max_rows: Optional[int] = None):
    print("Loading metadata CSV:", metadata_csv)
    df = pd.read_csv(metadata_csv)
    print("Rows in CSV:", len(df))
    pca = load_pca(pca_path)
    print("Loaded PCA:", pca)

    # Prepare output file
    header = ["dialogue_id","utt_id","text","label","valence","action","z_x","z_y","reply"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

    dialogue_history = {}
    processed = 0
    for idx, row in df.iterrows():
        if max_rows is not None and processed >= max_rows:
            break

        # adapt column names (try multiple common names)
        dlg = row.get("Dialogue_ID", row.get("dialogue_id", row.get("dialog_id", "dlg")))
        utt = row.get("Utterance_ID", row.get("utt_id", row.get("utt", idx)))
        text = row.get("text", row.get("utterance", ""))
        label = row.get("label", "")
        valence = float(row.get("valence", 0.0)) if "valence" in row.index else 0.0
        action = int(row.get("action", 0)) if "action" in row.index else 0
        state_path = row.get("state_path", "")

        if dlg not in dialogue_history:
            dialogue_history[dlg] = []

        # load z (safe)
        if isinstance(state_path, str) and state_path and os.path.exists(state_path):
            try:
                z = np.load(state_path).ravel()
            except Exception:
                z = np.zeros(pca.components_.shape[1], dtype=np.float32)
        else:
            z = np.zeros(pca.components_.shape[1], dtype=np.float32)

        # ensure z length matches PCA expected dim
        expected_dim = pca.components_.shape[1]
        if z.size < expected_dim:
            z2 = np.zeros(expected_dim, dtype=np.float32)
            z2[:z.size] = z
            z = z2
        elif z.size > expected_dim:
            z = z[:expected_dim]

        zproj = pca.transform(z.reshape(1,-1))[0]

        prompt = build_small_prompt(text, label, valence, zproj, action, dialogue_history[dlg])
        reply = call_openai_chat(prompt)

        # record
        dialogue_history[dlg].append(f"U: {text}\nA: {reply}")

        with open(out_csv, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([dlg, utt, text, label, valence, action, float(zproj[0]), float(zproj[1]), reply])

        processed += 1
        if processed % 500 == 0:
            print("Processed:", processed)

    print("Simulation finished. Wrote:", out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, default="data/processed/meld_text_audio_video_arcface_states.csv")
    parser.add_argument("--pca_path", type=str, default="models/pca_states_2.pkl")
    parser.add_argument("--out", type=str, default="simulation_outputs.csv")
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()
    simulate(args.metadata_csv, args.pca_path, args.out, max_rows=args.max_rows)
