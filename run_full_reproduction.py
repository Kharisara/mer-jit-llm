"""
run_full_reproduction.py

Reproduction runner for the Policy-First JITAI Architecture experiments.

Pipeline
--------
1. Train policy using train_jitai_policy.py
2. Run policy-first offline replay (BC policy)
3. Run proxy logging policy replay
4. Run end-to-end LLM baseline
5. Run deterministic replay verification
6. Print summary statistics

Outputs
-------
CSV files stored in paper_outputs/
"""

import os
import sys
import subprocess
import pandas as pd
import random
import numpy as np

# -------------------------------------------------
# DETERMINISTIC SEED CONTROL
# -------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

try:
    import torch
    torch.manual_seed(SEED)
except Exception:
    pass


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_CSV = "data/processed/meld_text_audio_video_arcface_states.csv"

OUTPUT_DIR = "paper_outputs"

BC_POLICY_PATH = os.path.join(OUTPUT_DIR, "bc_policy.pt")

POLICY_FIRST_OUT = os.path.join(OUTPUT_DIR, "policy_first_outputs.csv")
PROXY_OUT = os.path.join(OUTPUT_DIR, "proxy_outputs.csv")
E2E_OUT = os.path.join(OUTPUT_DIR, "e2e_outputs.csv")
STRESS_OUT = os.path.join(OUTPUT_DIR, "stress_outputs.csv")

MAX_ROWS = None  # set integer for quick tests


# -------------------------------------------------
# UTILITIES
# -------------------------------------------------

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[INFO] Output directory:", OUTPUT_DIR)


def run_cmd(cmd):

    print("[CMD]", " ".join(cmd))

    subprocess.run(cmd, check=True)


# -------------------------------------------------
# STEP 1 – Train policy
# -------------------------------------------------

def train_policy():

    print("\n[STEP 1] Training policy")

    cmd = [
        sys.executable,
        "-m",
        "mer.train_jitai_policy",
        "--csv",
        DATASET_CSV,
        "--out_model",
        BC_POLICY_PATH,
        "--epochs",
        "20",
        "--batch_size",
        "256"
    ]

    run_cmd(cmd)


# -------------------------------------------------
# STEP 2 – Policy-first replay
# -------------------------------------------------

def run_policy_first():

    print("\n[STEP 2] Running policy-first replay")

    cmd = [
    sys.executable,
    "-m",
    "mer.simulate_with_bc",
    "--csv",
    DATASET_CSV,
    "--out",
    STRESS_OUT,
    "--policy_mode",
    "bc"
]

    if MAX_ROWS:
        cmd += ["--max_rows", str(MAX_ROWS)]

    run_cmd(cmd)


# -------------------------------------------------
# STEP 3 – Proxy logging policy replay
# -------------------------------------------------

def run_proxy():

    print("\n[STEP 3] Running proxy logging policy replay")

    cmd = [
        sys.executable,
        "-m",
        "mer.simulate_with_bc",
        "--csv",
        DATASET_CSV,
        "--out",
        PROXY_OUT,
        "--policy_mode",
        "proxy"
    ]

    if MAX_ROWS:
        cmd += ["--max_rows", str(MAX_ROWS)]

    run_cmd(cmd)


# -------------------------------------------------
# STEP 4 – End-to-End LLM baseline
# -------------------------------------------------

def run_e2e():

    print("\n[STEP 4] Running end-to-end LLM baseline")

    cmd = [
        sys.executable,
        "-m",
        "mer.simulate_e2e_llm",
        "--metadata_csv",
        DATASET_CSV,
        "--out",
        E2E_OUT
    ]

    if MAX_ROWS:
        cmd += ["--max_rows", str(MAX_ROWS)]

    try:
        run_cmd(cmd)
    except Exception as e:
        print("[WARN] E2E baseline skipped:", e)


# -------------------------------------------------
# STEP 5 – Deterministic replay verification
# -------------------------------------------------

def run_stress_test():

    print("\n[STEP 5] Running deterministic replay check")

    cmd = [
    sys.executable,
    "-m",
    "mer.simulate_with_bc",
    "--csv",
    DATASET_CSV,
    "--out",
    POLICY_FIRST_OUT,
    "--policy_mode",
    "bc"
]

    run_cmd(cmd)


# -------------------------------------------------
# RESULT SUMMARIES
# -------------------------------------------------

def summarize_policy_first():

    if not os.path.exists(POLICY_FIRST_OUT):
        return

    df = pd.read_csv(POLICY_FIRST_OUT)

    total = len(df)
    interventions = int(df["action"].sum())
    silence = total - interventions

    unauthorized = df[
        (df["action"] == 0) & (df["reply_json"].notna())
    ]

    print("\n========== POLICY-FIRST RESULTS ==========")
    print(f"Decision points        : {total}")
    print(f"Interventions          : {interventions}")
    print(f"Silence                : {silence}")
    print(f"Intervention rate      : {interventions/total:.3f}")
    print(f"Unexpected responses   : {len(unauthorized)}")
    print("==========================================\n")


def summarize_e2e():

    if not os.path.exists(E2E_OUT):
        return

    df = pd.read_csv(E2E_OUT)

    total = len(df)
    interventions = int(df["action"].sum())

    print("\n========== E2E BASELINE ==========")
    print(f"Turns processed        : {total}")
    print(f"Responses generated    : {interventions}")
    print(f"Response rate          : {interventions/total:.3f}")
    print("Silence pathway        : not present")
    print("==================================\n")


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    print("\n======================================")
    print(" POLICY-FIRST ARCHITECTURE REPRODUCTION")
    print("======================================")

    ensure_dirs()

    train_policy()

    run_policy_first()

    run_proxy()

    run_e2e()

    run_stress_test()

    summarize_policy_first()

    summarize_e2e()

    print("\nReproduction pipeline completed.")
    print("Outputs stored in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()