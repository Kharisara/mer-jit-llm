# Policy-First JITAI Architecture for Adaptive Conversational Systems

This repository contains the experimental implementation used in the paper:

**“Policy-First Architectures for Offline Evaluation of Adaptive Conversational Systems”**

The project demonstrates a **policy-first conversational architecture** in which **intervention timing is controlled by an explicit decision policy before language generation**. The system is evaluated under **deterministic offline replay** using the MELD multimodal conversational dataset.

The goal of this implementation is **system-level validation**, not clinical deployment or therapeutic evaluation.

---

# Repository Overview

The system implements a modular conversational architecture consisting of:

1. **Multimodal affective state construction**
2. **Offline intervention policy**
3. **Policy-gated language generation**
4. **Deterministic offline replay evaluation**

Key architectural property:

```
Decision Policy → Authorization → Language Model Invocation
```

Language generation occurs **only after policy approval**, enabling explicit non-intervention and controlled intervention timing.

---

# Experimental Pipeline

The repository includes a complete reproducible pipeline:

1. Behavioural policy training
2. Policy-first offline replay evaluation
3. Proxy logging policy comparison
4. End-to-end LLM baseline simulation
5. Deterministic replay validation
6. Automatic metric summarization

All experiments operate on **fixed logged conversational data** under offline conditions.

---

# Project Structure

```
mer-jit-llm
│
├── run_full_reproduction.py
├── run_ipd_demo.py
├── README.md
│
├── mer/
│   ├── train_jitai_policy.py
│   ├── simulate_with_bc.py
│   ├── simulate_e2e_llm.py
│
├── data/
│   └── processed/
│       └── meld_text_audio_video_arcface_states.csv
│
└── paper_outputs/
```

---

# Installation

Create a Python virtual environment.

```
python -m venv .venv
```

Activate the environment.

**Windows**

```
.venv\Scripts\activate
```

Install dependencies.

```
pip install -r requirements.txt
```

---

# Reproducing the Experiments

The full experimental pipeline can be reproduced using a single command:

```
python run_full_reproduction.py
```

This script executes the complete experimental workflow:

1. Train the intervention policy
2. Run policy-first offline replay
3. Run proxy logging comparison
4. Run end-to-end LLM baseline
5. Perform deterministic replay validation
6. Generate summary statistics

Outputs are written to:

```
paper_outputs/
```

---

# Expected Outputs

After running the reproduction script, the following files will be generated:

```
paper_outputs/
    bc_policy.pt
    policy_first_outputs.csv
    proxy_outputs.csv
    e2e_outputs.csv
    stress_outputs.csv
```

Example reported metrics:

| Metric                       | Value  |
| ---------------------------- | ------ |
| Decision points              | 11,351 |
| Interventions                | 2,609  |
| Silence                      | 8,742  |
| Intervention rate            | ≈ 0.23 |
| Unauthorized LLM invocations | 0      |

These results confirm deterministic replay behaviour and policy-gated language generation.

---

# Demonstration Script

A lightweight demonstration pipeline is also provided:

```
python run_ipd_demo.py
```

This script runs a smaller experiment for quick verification of the policy-first architecture.

---

# Dataset

Experiments are conducted using the **MELD multimodal conversational dataset**.

The dataset provides:

* Text embeddings
* Audio feature embeddings
* Visual (ArcFace) facial embeddings
* Emotion labels

The dataset does **not contain real intervention actions or outcome variables**. Therefore, experiments evaluate **system behaviour under offline replay** rather than intervention effectiveness.

---

# Reproducibility Notes

All experiments operate under:

* deterministic offline replay
* fixed logged conversational data
* policy-gated language model invocation
* no online interaction

As a result, reported metrics represent **exact empirical quantities rather than stochastic estimates**.

---

# Limitations

This implementation is intended for **methodological validation** only.

Important limitations include:

* No clinical intervention data
* No outcome-based reward signals
* Binary intervention decision space
* Offline evaluation without user interaction

The system should **not be interpreted as a deployable mental health intervention system**.

---

# License

This repository is provided for **research and reproducibility purposes**.

---

# Contact

For questions regarding the experimental implementation or reproduction pipeline, please open an issue in the repository.
