# # mer/simulate_with_bc.py
# """
# Simulate dialogues using a Behavioral Cloning (BC) policy and optionally generate
# LLM replies (local or cloud). Writes a CSV with (dialogue_id, utt_id, text, label, valence,
# action, z_x, z_y, reply, model_used).

# Usage examples:
# # run locally (no LLM)
# python -m mer.simulate_with_bc --csv data/processed/splits/meld_meta_chunk02.csv --out simulation_local_chunk02.csv --no_llm

# # use local Ollama/text-generation-webui server
# python -m mer.simulate_with_bc --csv data/processed/splits/meld_meta_chunk02.csv --out simulation_local_chunk02.csv --use_local_llm --local_url http://localhost:11434 --local_model llama-3.1-8b-instant --max_tokens 40

# # fallback to cloud (Groq) if local absent (requires $env:GROQ_API_KEY)
# python -m mer.simulate_with_bc --csv data/processed/splits/meld_meta_chunk02.csv --out simulation_cloud_sample.csv --model llama-3.1-8b-instant --max_tokens 80
# """
# import os
# import argparse
# import csv
# import time
# import json
# from typing import Optional

# import numpy as np
# import pandas as pd
# import joblib
# import requests

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # --------------------------
# # Simple label -> valence map
# # --------------------------
# LABEL2VALENCE = {
#     "angry": -1.0,
#     "sad": -1.0,
#     "neutral": 0.0,
#     "happy": 1.0,
# }

# def label_to_valence(label: str) -> float:
#     return LABEL2VALENCE.get(str(label).lower(), 0.0)

# # --------------------------
# # BCPolicy (must match training code)
# # --------------------------
# class BCPolicy(nn.Module):
#     def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_actions: int = 2):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.out = nn.Linear(hidden_dim, num_actions)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         logits = self.out(x)
#         return logits

# # --------------------------
# # Local LLM caller
# # --------------------------
# def call_local_llm(prompt: str, model: str, max_tokens: int = 80, base_url: str = "http://localhost:11434", timeout: int = 30) -> Optional[str]:
#     """
#     Try common local endpoints (Ollama-style and text-generation-webui).
#     Return string on success, None on failure.
#     """
#     headers = {"Content-Type": "application/json"}
#     # Ollama-style /api/generate
#     try:
#         url = base_url.rstrip("/") + "/api/generate"
#         payload = {"model": model, "prompt": prompt, "max_tokens": int(max_tokens)}
#         resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
#         if resp.status_code == 200:
#             try:
#                 j = resp.json()
#                 # Various shapes: completion, text, choices...
#                 if isinstance(j, dict):
#                     if "completion" in j:
#                         return str(j["completion"]).strip()
#                     if "text" in j:
#                         return str(j["text"]).strip()
#                     if "choices" in j and len(j["choices"])>0:
#                         c = j["choices"][0]
#                         if isinstance(c, dict) and "message" in c and "content" in c["message"]:
#                             return str(c["message"]["content"]).strip()
#                         if isinstance(c, dict) and "text" in c:
#                             return str(c["text"]).strip()
#                 return resp.text.strip()
#             except Exception:
#                 return resp.text.strip()
#         # else continue
#     except Exception as e:
#         print(f"[LOCAL_LLM] Ollama-style endpoint error: {e}")

#     # text-generation-webui /api/v1/generate
#     try:
#         url2 = base_url.rstrip("/") + "/api/v1/generate"
#         payload2 = {"model": model, "input": prompt, "max_new_tokens": int(max_tokens)}
#         resp2 = requests.post(url2, headers=headers, json=payload2, timeout=timeout)
#         if resp2.status_code == 200:
#             try:
#                 j2 = resp2.json()
#                 if isinstance(j2, dict) and "results" in j2 and len(j2["results"])>0:
#                     return str(j2["results"][0].get("text","")).strip()
#                 if isinstance(j2, dict) and "text" in j2:
#                     return str(j2["text"]).strip()
#                 return resp2.text.strip()
#             except Exception:
#                 return resp2.text.strip()
#     except Exception as e:
#         print(f"[LOCAL_LLM] text-generation-webui endpoint error: {e}")

#     return None

# # --------------------------
# # Cloud LLM caller (Groq OpenAI-compatible)
# # --------------------------
# def call_cloud_llm_groq(prompt: str, model: str, max_tokens: int = 80, temperature: float = 0.7, timeout: int = 30) -> Optional[str]:
#     """
#     Try Groq's OpenAI-compatible chat completions endpoint if GROQ_API_KEY is present.
#     Returns text or None.
#     """
#     gkey = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY")
#     if not gkey:
#         return None

#     url = "https://api.groq.com/openai/v1/chat/completions"
#     headers = {"Authorization": f"Bearer {gkey}", "Content-Type": "application/json"}
#     # single message style user -> system minimal
#     payload = {
#         "model": model,
#         "messages": [
#             {"role":"system","content":"You are a concise empathic assistant for short supportive replies."},
#             {"role":"user","content": prompt}
#         ],
#         "temperature": float(temperature),
#         "max_tokens": int(max_tokens),
#     }
#     try:
#         resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
#         if resp.status_code == 200:
#             j = resp.json()
#             # OpenAI-like response
#             if "choices" in j and len(j["choices"])>0:
#                 c = j["choices"][0]
#                 # chat style
#                 if "message" in c and "content" in c["message"]:
#                     return str(c["message"]["content"]).strip()
#                 if "text" in c:
#                     return str(c["text"]).strip()
#             # else fallback to raw text
#             return resp.text.strip()
#         else:
#             # if rate-limited or other, print warning
#             try:
#                 err = resp.json()
#             except Exception:
#                 err = resp.text
#             print(f"[CLOUD_LLM] Groq returned status {resp.status_code}: {err}")
#             return None
#     except Exception as e:
#         print(f"[CLOUD_LLM] Request failed: {e}")
#         return None

# # --------------------------
# # Stub reply helper
# # --------------------------
# def make_stub_reply(user_text: str, label: str = "neutral") -> str:
#     # short, safe, generic fallback reply (could be adapted per label)
#     if str(label).lower() in ("sad", "angry"):
#         return "Thanks for telling me — that sounds hard. Try one slow breath now and notice what you feel. You're not alone."
#     if str(label).lower() == "happy":
#         return "That's great to hear — thank you for sharing. If you'd like, tell me more about what went well."
#     return "Thanks for sharing that — I hear you. Take a slow breath and notice what feels most present for you right now."

# # --------------------------
# # Prompt builder
# # --------------------------
# def build_small_prompt(text: str, label: str, valence: float, zproj: tuple, recent_history: list, action: int) -> str:
#     """
#     Very compact prompt including valence and 2D projection and recent history.
#     Keeps outputs short and directive for action==1.
#     """
#     hx = ""
#     if recent_history:
#         hx = "Recent: " + " | ".join(recent_history[-3:])
#     px = f"User utterance: \"{text}\"\nDetected emotion: {label} (valence={valence:.2f})\nAffective projection: x={zproj[0]:.3f}, y={zproj[1]:.3f}\n{hx}\nPolicy action:{action}\n"
#     if action == 1:
#         px += "Task: produce exactly 2 short sentences. Sentence1: empathic validation. Sentence2: Action: provide a single explicit, practical step starting with 'Action:'.\n"
#     else:
#         px += "Task: produce 1-2 short empathic sentences (no action suggested).\n"
#     px += "Be concise; do not provide diagnoses."
#     return px

# # --------------------------
# # Main simulate function
# # --------------------------
# def simulate(
#     csv_path: str,
#     out_path: str,
#     pca_path: str = "models/pca_states_2.pkl",
#     bc_model_path: str = "models/jitai_policy_bc.pt",
#     max_rows: int = -1,
#     preferred_model: str = "llama-3.1-8b-instant",
#     max_tokens: int = 60,
#     use_local_llm: bool = False,
#     local_url: str = "http://localhost:11434",
#     local_model: str = "llama-3.1-8b-instant",
#     no_llm: bool = False,
# ):
#     # Load metadata CSV
#     print("Loading metadata CSV:", csv_path)
#     df = pd.read_csv(csv_path)
#     total_rows = len(df)
#     print("Rows:", total_rows)
#     if max_rows is None or max_rows <= 0:
#         max_rows = total_rows
#     else:
#         max_rows = min(max_rows, total_rows)
#     print("Processing up to:", max_rows)

#     # Load PCA
#     if os.path.exists(pca_path):
#         pca = joblib.load(pca_path)
#         print("Loaded PCA:", pca)
#     else:
#         pca = None
#         print("[WARN] PCA file not found:", pca_path)

#     # Load BC policy weights
#     device = torch.device("cpu")
#     model = BCPolicy(input_dim=512, hidden_dim=256, num_actions=2).to(device)
#     if os.path.exists(bc_model_path):
#         try:
#             st = torch.load(bc_model_path, map_location=device)
#             # Expect state_dict; handle direct state_dict or full model dict
#             if isinstance(st, dict) and any(k.startswith("fc1") or k.startswith("fc1.") for k in st.keys()):
#                 model.load_state_dict(st)
#             else:
#                 try:
#                     model.load_state_dict(st)
#                 except Exception:
#                     # if saved as model.state_dict() maybe wrapped; try to use directly
#                     model.load_state_dict(st)
#             print("Loaded BC policy:", bc_model_path)
#         except Exception as e:
#             print("[WARN] Failed to load BC policy:", e)
#     else:
#         print("[WARN] BC policy not found at:", bc_model_path)

#     # Prepare output CSV (header)
#     os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
#     fh = open(out_path, "w", encoding="utf-8", newline="")
#     writer = csv.writer(fh)
#     writer.writerow(["dialogue_id","utt_id","text","label","valence","action","z_x","z_y","reply","model_used"])

#     recent_history_by_dialogue = {}

#     processed = 0
#     for idx in range(max_rows):
#         row = df.iloc[idx]
#         dialogue_id = int(row.get("Dialogue_ID", -1))
#         utt_id = int(row.get("Utterance_ID", -1))
#         text = str(row.get("text",""))
#         label = str(row.get("label","neutral"))
#         valence = float(row.get("valence", label_to_valence(label)))
#         state_path = row.get("state_path", None)

#         # track recent history
#         if dialogue_id not in recent_history_by_dialogue:
#             recent_history_by_dialogue[dialogue_id] = []
#         recent_history_by_dialogue[dialogue_id].append(text)

#         # Load state vector
#         state_np = None
#         if isinstance(state_path, str) and os.path.exists(state_path):
#             try:
#                 state_np = np.load(state_path)
#                 state_np = state_np.reshape(-1).astype(np.float32)
#             except Exception as e:
#                 print(f"[WARN] failed to load state at {state_path}: {e}")
#                 state_np = None

#         if state_np is None:
#             # skip if no state available (or optionally use zeros)
#             state_np = np.zeros((512,), dtype=np.float32)

#         # Compute z_proj (2D) if PCA present
#         if pca is not None:
#             try:
#                 zproj = pca.transform(state_np.reshape(1,-1))[0]
#             except Exception:
#                 # if pca expects different dim, fallback to zeros
#                 zproj = np.array([0.0, 0.0], dtype=np.float32)
#         else:
#             zproj = np.array([0.0, 0.0], dtype=np.float32)

#         # Determine action from BC policy
#         with torch.no_grad():
#             s_t = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
#             logits = model(s_t)
#             action = int(torch.argmax(logits, dim=-1).item())

#         # Build prompt
#         prompt = build_small_prompt(text=text, label=label, valence=valence, zproj=(float(zproj[0]), float(zproj[1])), recent_history=recent_history_by_dialogue[dialogue_id], action=action)

#         # Decide reply & model_used
#         reply = None
#         model_used = ""

#         # 1) If user requested local LLM, try it first
#         if use_local_llm and not no_llm:
#             try:
#                 local_resp = call_local_llm(prompt=prompt, model=local_model, max_tokens=max_tokens, base_url=local_url)
#                 if local_resp is not None and str(local_resp).strip() != "":
#                     reply = str(local_resp).strip()
#                     model_used = f"local:{local_model}"
#                 else:
#                     print("[LOCAL_LLM] empty response, falling back")
#             except Exception as e:
#                 print(f"[LOCAL_LLM] error: {e}")

#         # 2) If still no reply and cloud allowed, try Groq
#         if reply is None and not no_llm:
#             cloud_resp = call_cloud_llm_groq(prompt=prompt, model=preferred_model, max_tokens=max_tokens)
#             if cloud_resp is not None and str(cloud_resp).strip() != "":
#                 reply = str(cloud_resp).strip()
#                 model_used = preferred_model
#             else:
#                 # cloud failed or not configured -> will fallback to stub later
#                 pass

#         # 3) If no_llm flag, or if both LLMs failed, use stub reply
#         if no_llm or reply is None or str(reply).strip() == "":
#             reply = make_stub_reply(text, label=label)
#             model_used = model_used or "stub"

#         # Write row
#         writer.writerow([dialogue_id, utt_id, text, label, valence, action, float(zproj[0]), float(zproj[1]), reply, model_used])
#         processed += 1

#         # print progress occasionally
#         if processed % 10 == 0 or processed == max_rows:
#             print(f"Processed {processed}/{max_rows} (dialogue {dialogue_id} utt {utt_id})")

#     fh.close()
#     print("Simulation complete ->", out_path)

# # --------------------------
# # CLI
# # --------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv", type=str, default="data/processed/meld_text_audio_video_arcface_states.csv")
#     parser.add_argument("--out", type=str, default="simulation_outputs.csv")
#     parser.add_argument("--max_rows", type=int, default=-1, help="Max rows to process (-1 for all)")
#     parser.add_argument("--model", type=str, default="llama-3.1-8b-instant", help="Preferred cloud model name (Groq)")
#     parser.add_argument("--max_tokens", type=int, default=60, help="Max tokens for LLM responses")
#     parser.add_argument("--use_local_llm", action="store_true", help="Try a local LLM server before cloud")
#     parser.add_argument("--local_url", type=str, default="http://localhost:11434", help="Local LLM base URL")
#     parser.add_argument("--local_model", type=str, default="llama-3.1-8b-instant", help="Local model name")
#     parser.add_argument("--no_llm", action="store_true", help="Disable LLM calls and use stub replies")
#     args = parser.parse_args()

#     simulate(
#         csv_path=args.csv,
#         out_path=args.out,
#         pca_path="models/pca_states_2.pkl",
#         bc_model_path="models/jitai_policy_bc.pt",
#         max_rows=args.max_rows,
#         preferred_model=args.model,
#         max_tokens=args.max_tokens,
#         use_local_llm=args.use_local_llm,
#         local_url=args.local_url,
#         local_model=args.local_model,
#         no_llm=args.no_llm,
#     )

# if __name__ == "__main__":
#     main()

"""
mer/simulate_with_bc.py

Rewritten simulate script with:
 - local LLM support (Ollama / text-gen-webui style at --local_url)
 - Groq cloud support with candidate_models and exponential backoff retries
 - auto-fallback between models (tries local → candidate groq models)
 - strict JSON output requirement (prompt asks model to return ONE JSON object)
 - --no_llm mode that returns a safe, deterministic stub reply (useful for offline runs)

USAGE examples (from project root):
 python -m mer.simulate_with_bc --max_rows 100 --out simulation_local_chunk01.csv \
    --use_local_llm --local_url http://localhost:11434 --local_model gemma3:1b --max_tokens 40

 python -m mer.simulate_with_bc --max_rows 100 --out simulation_bc_groq_sample.csv \
    --model llama-3.1-8b-instant

 Notes:
  - Set GROQ_API_KEY env var if using Groq cloud.
  - The script *requires* that the LLM returns a single JSON object in its full-text reply, for example:
      {"sentences": ["I hear you.", "Try a 5-breath grounding."], "safety": "ok"}
    The script will try to extract the first JSON object found in the model response.

Tweak the prompt text below if you want a different reply format.

"""

import os
import csv
import json
import time
import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests
import torch

# default logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("simulate_with_bc")

# -----------------------
# Simple utility: safe JSON extraction
# -----------------------

def extract_first_json(s: str) -> Optional[dict]:
    """Find the first {...} JSON object in string s and parse it.
    Tolerant: trims before/after, returns None on failure.
    """
    if not isinstance(s, str):
        return None
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    cand = s[start : end + 1]
    try:
        return json.loads(cand)
    except Exception:
        # Try a brute-force balanced-brace scan (first balanced {})
        depth = 0
        for i, ch in enumerate(s):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except Exception:
                        return None
        return None

# -----------------------
# LLM callers
# -----------------------

def call_local_llm(prompt: str, local_url: str, local_model: str, max_tokens: int = 80, timeout: int = 20) -> Optional[str]:
    """Call a local LLM server (Ollama / text-generation-webui). This is kept permissive:
    - tries POST to common endpoints: /api/generate, /v1/generate, /api/completions
    - body varies slightly depending on endpoint
    Returns the raw text if successful, else None.
    """
    endpoints = ["/api/generate", "/v1/generate", "/api/completions", "/v1/completions"]
    headers = {"Content-Type": "application/json"}

    payloads = [
        # Ollama-style compat
        {"model": local_model, "prompt": prompt, "max_tokens": max_tokens, "stream": False},
        # text-generation-webui style
        {"model": local_model, "inputs": prompt, "max_new_tokens": max_tokens},
        # generic OpenAI-like
        {"model": local_model, "messages": [{"role":"user","content":prompt}], "max_tokens": max_tokens},
    ]

    for ep in endpoints:
        url = local_url.rstrip("/") + ep
        for payload in payloads:
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            except Exception as e:
                logger.debug(f"Local LLM call to {url} failed: {e}")
                continue
            if r.status_code != 200:
                logger.debug(f"Local LLM {url} status {r.status_code} -> {r.text[:200]}")
                continue
            # Try to parse a sensible reply string
            try:
                j = r.json()
            except Exception:
                # fallback to raw text
                return r.text
            # Many servers return nested structures; try common fields
            for key in ("text", "response", "output", "generated_text", "choices"):
                if key in j:
                    val = j[key]
                    # choices may be list of dicts
                    if isinstance(val, list):
                        # try to join text fields
                        texts = []
                        for c in val:
                            if isinstance(c, dict):
                                for k2 in ("text", "response", "output", "message", "content"):
                                    if k2 in c:
                                        texts.append(str(c[k2]))
                                        break
                            else:
                                texts.append(str(c))
                        return "".join(texts)
                    else:
                        return str(val)
            # Ollama sometimes returns streaming chunks joined by newlines; try to extract 'response' fields
            if isinstance(j, dict) and any(isinstance(v, dict) and "response" in v for v in j.values()):
                # flatten responses
                parts = []
                def collect(x):
                    if isinstance(x, dict):
                        if "response" in x:
                            parts.append(str(x["response"]))
                        for v in x.values():
                            collect(v)
                    elif isinstance(x, list):
                        for e in x:
                            collect(e)
                collect(j)
                if parts:
                    return "".join(parts)
            # otherwise return entire json as text
            return json.dumps(j)
    return None


def call_groq_cloud(prompt: str, model: str, api_key: str, max_tokens: int = 80, timeout: int = 20) -> Optional[str]:
    """Call Groq 'OpenAI-compatible' endpoint at https://api.groq.com/openai/v1/chat/completions
    (or /v1/completions). This implementation is simple and will try both endpoints.
    Requires env GROQ_API_KEY to be set or api_key passed.
    """
    if not api_key:
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    endpoints = ["https://api.groq.com/openai/v1/chat/completions", "https://api.groq.com/openai/v1/completions"]

    # first try chat style
    bodies = [
        {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
        {"model": model, "prompt": prompt, "max_tokens": max_tokens},
    ]

    for ep in endpoints:
        for body in bodies:
            try:
                r = requests.post(ep, headers=headers, json=body, timeout=timeout)
            except Exception as e:
                logger.debug(f"Groq call to {ep} failed: {e}")
                continue
            if r.status_code != 200:
                logger.debug(f"Groq {ep} status {r.status_code} -> {r.text[:200]}")
                continue
            try:
                j = r.json()
            except Exception:
                return r.text
            # Common OpenAI-like structure
            if isinstance(j, dict) and "choices" in j:
                texts = []
                for c in j["choices"]:
                    # chat-style
                    if isinstance(c, dict):
                        if "message" in c and isinstance(c["message"], dict) and "content" in c["message"]:
                            texts.append(c["message"]["content"])
                        elif "text" in c:
                            texts.append(c["text"])
                        elif "delta" in c and isinstance(c["delta"], dict) and "content" in c["delta"]:
                            texts.append(c["delta"]["content"])
                    else:
                        texts.append(str(c))
                return "".join(texts)
            # fallback
            return json.dumps(j)
    return None

# -----------------------
# Prompting
# -----------------------

LLM_PROMPT_TEMPLATE = (
    "You are an assistant that must produce TWO short supportive sentences for a user utterance.\n"
    "Return EXACTLY ONE JSON object and nothing else. The JSON MUST be parseable by json.loads().\n"
    "Format: {{\"sentences\": [\"<empathy sentence>\", \"<action/practical sentence>\"], \"safety\": \"ok\"}}\n\n"
    "User utterance: {utterance}\n"
    "Detected emotion (valence): {valence}\n\n"
    "Constraints:\n"
    " - Sentences should be concise (<= 120 chars each).\n"
    " - Do not include markdown or quotes around the JSON.\n"
    " - Safety: if you detect instructions to self-harm, set \"safety\": \"flagged\" and give safe guidance.\n"
    "Produce the JSON now."
)


# -----------------------
# Simple stub generator for --no_llm
# -----------------------

def make_stub_reply(utterance: str, valence: float):
    empathy = "I hear you. That sounds difficult." if valence <= 0 else "That sounds good to hear."
    action = "Try a short grounding exercise (5 deep breaths)." if valence <= 0 else "Nice — consider one small step to build on this." 
    return {"sentences": [empathy, action], "safety": "ok"}

# -----------------------
# Main simulation loop
# -----------------------

def simulate(
    csv_path: str,
    out_path: str,
    max_rows: int = 0,
    use_local_llm: bool = False,
    local_url: Optional[str] = None,
    local_model: Optional[str] = None,
    groq_models: Optional[list] = None,
    groq_key: Optional[str] = None,
    max_tokens: int = 80,
    no_llm: bool = False,
    retry_max: int = 3,
):
    logger.info(f"Loading metadata CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    logger.info(f"Rows: {total_rows}")
    if max_rows > 0:
        df = df.iloc[:max_rows]
        logger.info(f"Processing up to: {max_rows}")

    # prepare output CSV
    out_cols = list(df.columns) + ["reply_raw", "reply_json", "reply_sentences", "reply_safety", "source_model"]
    writer = csv.DictWriter(open(out_path, "w", newline='', encoding='utf-8'), fieldnames=out_cols)
    writer.writeheader()

    # iterate
    for i, row in df.reset_index(drop=True).iterrows():
        text = str(row.get("text", ""))
        label = str(row.get("label", ""))
        valence = float(row.get("valence", 0.0)) if "valence" in row else 0.0

        reply_raw = None
        reply_json = None
        used_model = None

        if no_llm:
            reply_json = make_stub_reply(text, valence)
            reply_raw = json.dumps(reply_json)
            used_model = "NO_LLM_STUB"
        else:
            prompt = LLM_PROMPT_TEMPLATE.format(utterance=text, valence=valence)

            # 1) try local LLM (if requested)
            if use_local_llm and local_url and local_model:
                for attempt in range(1, retry_max + 1):
                    try:
                        reply_raw = call_local_llm(prompt, local_url, local_model, max_tokens=max_tokens)
                        if reply_raw:
                            reply_json = extract_first_json(reply_raw)
                            if reply_json is not None:
                                used_model = f"local:{local_model}"
                                break
                            else:
                                logger.debug("Local LLM reply not parseable as JSON — will retry/continue to fallback")
                        else:
                            logger.debug("Local LLM returned no text")
                    except Exception as e:
                        logger.debug(f"Local LLM attempt {attempt} failed: {e}")
                    # simple backoff
                    time.sleep(0.5 * attempt)
                # if local produced raw but not JSON, keep raw for inspection and fall back

            # 2) try Groq cloud candidate models (auto-fallback)
            if reply_json is None and groq_models and groq_key:
                for model in groq_models:
                    for attempt in range(1, retry_max + 1):
                        try:
                            reply_raw = call_groq_cloud(prompt, model=model, api_key=groq_key, max_tokens=max_tokens)
                            if reply_raw:
                                reply_json = extract_first_json(reply_raw)
                                if reply_json is not None:
                                    used_model = f"groq:{model}"
                                    break
                                else:
                                    logger.debug(f"Groq model {model} reply not JSON-parseable")
                            else:
                                logger.debug(f"Groq model {model} returned no text")
                        except Exception as e:
                            logger.debug(f"Groq {model} attempt {attempt} failed: {e}")
                        # exponential backoff
                        time.sleep((2 ** (attempt - 1)) * 0.5)
                    if reply_json is not None:
                        break

            # 3) final fallback: if still no JSON, use stub but record raw
            if reply_json is None:
                if reply_raw is None:
                    # nothing at all returned — safe stub
                    reply_json = make_stub_reply(text, valence)
                    reply_raw = json.dumps(reply_json)
                    used_model = "FALLBACK_STUB"
                else:
                    # raw returned but couldn't parse
                    used_model = "RAW_NONJSON"
                    # include raw in reply_json for inspection
                    reply_json = {"sentences": [reply_raw[:120]], "safety": "unknown", "_raw": reply_raw}

        # write out row
        out_row = row.to_dict()
        out_row["reply_raw"] = reply_raw
        out_row["reply_json"] = json.dumps(reply_json, ensure_ascii=False) if reply_json is not None else ""
        out_row["reply_sentences"] = json.dumps(reply_json.get("sentences")) if isinstance(reply_json, dict) and "sentences" in reply_json else ""
        out_row["reply_safety"] = reply_json.get("safety") if isinstance(reply_json, dict) else ""
        out_row["source_model"] = used_model
        writer.writerow(out_row)

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"Processed {i+1}/{len(df)}")

    logger.info(f"Simulation complete -> {out_path}")

# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/processed/meld_text_audio_video_arcface_states.csv")
    parser.add_argument("--out", type=str, default="simulation_outputs_local.csv")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--use_local_llm", action="store_true")
    parser.add_argument("--local_url", type=str, default="http://localhost:11434")
    parser.add_argument("--local_model", type=str, default="gemma3:1b")
    parser.add_argument("--model", type=str, default=None, help="preferred groq model (single)")
    parser.add_argument("--candidate_models", type=str, default="llama-3.1-8b-instant,llama-3.3-70b-versatile", help="comma separated candidate groq models")
    parser.add_argument("--max_tokens", type=int, default=80)
    parser.add_argument("--no_llm", action="store_true", help="Do not call any LLM; use deterministic stub replies")
    parser.add_argument("--retry_max", type=int, default=3, help="Number of attempts per model")

    args = parser.parse_args()

    groq_key = os.environ.get("GROQ_API_KEY")
    candidates = []
    if args.model:
        candidates = [args.model]
    else:
        candidates = [m.strip() for m in args.candidate_models.split(",") if m.strip()]

    simulate(
        csv_path=args.csv,
        out_path=args.out,
        max_rows=args.max_rows,
        use_local_llm=args.use_local_llm,
        local_url=args.local_url,
        local_model=args.local_model,
        groq_models=candidates,
        groq_key=groq_key,
        max_tokens=args.max_tokens,
        no_llm=args.no_llm,
        retry_max=args.retry_max,
    )

if __name__ == "__main__":
    main()
