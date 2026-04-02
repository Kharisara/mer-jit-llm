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

Experiments are conducted using the MELD dataset (Multimodal EmotionLines Dataset), which is publicly available at:

https://affective-meld.github.io/

Due to licensing and size constraints, the dataset and its processed representations are not included in this repository.

ataset

Experiments are conducted using the MELD dataset (Multimodal EmotionLines Dataset), available at:

https://affective-meld.github.io/

Due to dataset usage terms and size constraints, the dataset and its processed representations are not included in this repository.

# Dataset Preparation
Download the dataset from:
https://affective-meld.github.io/

Place the dataset in:
data/raw/

Prepare the dataset using the provided scripts:
python reorganize_meld_frames.py
python extract_tav_context_states.py
python precompute_meld_video_embeddings.py

These steps prepare the multimodal inputs required for training and evaluation.
Refer to the individual scripts for configuration details.

# Notes
Full reproduction of experimental results requires access to the MELD dataset
Ensure the dataset is correctly prepared before running the pipeline
The repository provides all necessary code for evaluation, but not the dataset itself
---

# License

This code is released for research reproducibility purposes.