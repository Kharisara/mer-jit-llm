# Policy-First Architecture for Offline Validation of Emotion-Aware Conversational AI

This repository contains the reproducibility pipeline for the paper:

**A Policy-First Architecture for Offline Systems Validation of Emotion-Aware Conversational AI**

The code reproduces the offline evaluation experiments reported in the paper using deterministic replay over the MELD dataset.

---

# Requirements

Python 3.10+

It is recommended to run the code inside a virtual environment.

Create and activate a virtual environment:

python -m venv .venv

Activate (Windows):

.venv\Scripts\activate

Activate (Linux/Mac):

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

# Running the Full Reproduction Pipeline

All experiments can be reproduced using the following command:

python run_full_reproduction.py

(Alternatively, on systems with `make` installed)

make reproduce

---

# Pipeline Steps

The reproduction pipeline performs the following steps automatically:

1. Train the behavioural cloning policy
2. Run policy-first offline replay
3. Run proxy logging policy replay
4. Run end-to-end LLM baseline simulation
5. Run deterministic replay verification
6. Produce summary statistics

---

# Output Files

All generated outputs are stored in:

paper_outputs/

The main outputs are:

- `policy_first_outputs.csv` — policy-first architecture replay
- `proxy_outputs.csv` — proxy logging baseline
- `e2e_outputs.csv` — end-to-end LLM baseline
- `stress_outputs.csv` — deterministic replay verification

---

# Deterministic Reproducibility

All experiments use fixed random seeds and deterministic offline replay ordering to ensure identical results across runs.

---

# Dataset

Experiments are conducted using processed MELD state representations included in:

data/processed/

---

# License

This code is released for research reproducibility purposes.