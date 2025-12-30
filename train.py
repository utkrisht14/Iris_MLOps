import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from model import IrisMLP


FEATURE_COLS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET_COL = "Species"

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_model(data_path: str, artifacts_dir: str, epochs: int, batch_size: int, lr:float, seed:int) -> None:
    set_seed(seed)

    df = pd.read_csv(data_path)
    # Basic clean up and validation
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURE_COLS].astype(float).values
    y_raw = df[TARGET_COL].astype(str).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw).astype("int64")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train).astype("float32")
    X_val_scaled = scaler.transform(X_val).astype("float32")

    train_ds = TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu") # Keep it simple
    model = IrisMLP(input_dim=4, hidden_dim=32, num_classes=len(label_encoder.classes_)).to(device)

    # Set the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Set the model to training mode
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation accuracy
        # Set the model to validation mode
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / max(total, 1)
        avg_loss = running_loss / max(len(train_loader), 1)

        print(f"Epoch {epoch:02d} / {epochs} | loss: {avg_loss:.4f} | val_ac: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    os.makedirs(artifacts_dir, exist_ok=True)

    # Save torch model
    model_path = os.path.join(artifacts_dir, "model.pth")

    # Save torch model
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": 4,
            "hidden_dim": 32,
            "num_classes": len(label_encoder.classes_),
        },
        model_path,
    )

    # Save scaler & label
    with open(os.path.join(artifacts_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler ,f)

    with open(os.path.join(artifacts_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder ,f)


    print(f"\n Saved artifacts")
    print(f"- {model_path}")
    print(f"- {os.path.join(artifacts_dir, 'scaler.pkl')}")
    print(f"- {os.path.join(artifacts_dir, 'label_encoder.pkl')}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/iris.csv")
    parser.add_argument("--artifacts", type=str, default="./artifacts/")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_model(data_path=args.data, artifacts_dir=args.artifacts, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)

if __name__ == "__main__":
    main()

    







