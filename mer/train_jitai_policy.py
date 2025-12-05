# mer/train_jitai_policy.py

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score  # <-- FIXED: added accuracy_score

from .env_meld_jitai import MELDJITAITransitionDataset


class JITAIPolicyNet(nn.Module):
    """
    Simple MLP policy:
      input: state (e.g., 512-d from TAV+context)
      output: logits over 2 actions (0 = no intervention, 1 = intervene)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, num_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, state):
        return self.net(state)  # [B, num_actions]


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)

    csv_path = os.path.join(
        project_root, "data", "processed", "meld_text_audio_video_arcface_states.csv"
    )
    ckpt_dir = os.path.join(project_root, "checkpoints_jitai")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "jitai_policy_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("CSV:", csv_path)

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------
    # TRAIN on train + dev transitions (we just want more data)
    train_ds = MELDJITAITransitionDataset(
        csv_path=csv_path,
        splits=("train", "dev"),
        device=str(device),
    )
    # VALIDATE on test transitions
    dev_ds = MELDJITAITransitionDataset(
        csv_path=csv_path,
        splits=("test",),
        device=str(device),
    )

    if train_ds.state_dim == 0 or len(train_ds) == 0:
        raise RuntimeError(
            "No transitions found for train splits. "
            "Check that state_path is populated and extract_tav_context_states.py ran correctly."
        )

    print("Train transitions:", len(train_ds))
    print("Dev transitions:  ", len(dev_ds))
    print("State dim:        ", train_ds.state_dim)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=8, shuffle=False)

    # --------------------------------------------------------
    # Model + optimizer
    # --------------------------------------------------------
    model = JITAIPolicyNet(state_dim=train_ds.state_dim)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 20
    best_dev_f1 = 0.0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            states = batch["state"].to(device)   # [B, state_dim]
            actions = batch["action"].to(device) # [B]

            logits = model(states)               # [B, 2]
            loss = loss_fn(logits, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"Train loss: {avg_loss:.4f}")

        # ----------------------------------------------------
        # Dev evaluation
        # ----------------------------------------------------
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dev_loader:
                states = batch["state"].to(device)
                actions = batch["action"].to(device)

                logits = model(states)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(actions.cpu().numpy().tolist())

        if len(all_labels) == 0:
            print("WARNING: No dev transitions; skipping dev evaluation.")
            continue

        dev_acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
        dev_f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"Dev action accuracy: {dev_acc:.4f}")
        print(f"Dev action macro F1: {dev_f1:.4f}")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": train_ds.state_dim,
                },
                best_ckpt_path,
            )
            print(f"New best policy saved → {best_ckpt_path}")

    print("\nTraining complete.")
    print("Best Dev action macro F1:", best_dev_f1)

    # --------------------------------------------------------
    # Final TEST evaluation using best checkpoint
    # --------------------------------------------------------
    print("\nLoading best policy for TEST evaluation...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    policy = JITAIPolicyNet(state_dim=best_ckpt["state_dim"])
    policy.load_state_dict(best_ckpt["model_state_dict"])
    policy.to(device)
    policy.eval()

    test_ds = MELDJITAITransitionDataset(
        csv_path=csv_path,
        splits=("test",),
        device=str(device),
    )
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            s = batch["state"].to(device)
            a_true = batch["action"].cpu().numpy().tolist()

            logits = policy(s)
            a_pred = logits.argmax(dim=-1).cpu().numpy().tolist()

            all_labels.extend(a_true)
            all_preds.extend(a_pred)

    print(f"Test action accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Test action macro F1: {f1_score(all_labels, all_preds, average='macro'):.4f}")


if __name__ == "__main__":
    main()
