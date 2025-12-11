# tools/analyze_simulation.py
import pandas as pd
import numpy as np

df = pd.read_csv("simulation_outputs.csv")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# action distribution
print("\nAction distribution:")
print(df['action'].value_counts(dropna=False))

# supportive fraction (simple keyword heuristic)
keywords = ["sorry","i'm sorry","that sounds","i understand","that must be","it makes sense","try","consider","grounding"]
df['supportive'] = df['reply'].astype(str).str.lower().apply(lambda s: any(k in s for k in keywords))
print("\nSupportive fraction:", df['supportive'].mean())

# z_proj ranges
print("\nZ-proj ranges:")
print("x min/max/mean:", df['z_x'].min(), df['z_x'].max(), df['z_x'].mean())
print("y min/max/mean:", df['z_y'].min(), df['z_y'].max(), df['z_y'].mean())

# show a few low-valence examples (if valence exists)
if 'valence' in df.columns:
    low = df[df['valence'] <= -0.5].sample(n=min(5, len(df[df['valence'] <= -0.5])), random_state=42)
    print("\nSample low-valence replies:")
    for _, r in low.iterrows():
        print("----")
        print("dialogue:", r['dialogue_id'], "utt:", r['utt_id'], "valence:", r['valence'])
        print("text:", r['text'])
        print("reply:", r['reply'])
else:
    print("\nNo 'valence' column found; skipping valence-based samples.")
