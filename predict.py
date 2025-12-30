import os
import numpy as np
import torch
import pickle
from model import IrisMLP

FEATURE_COLS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

class IrisPredictor:
    def __init__(self, artifacts_dir: str = "artifacts"):
        model_path = os.path.join(artifacts_dir, "model.pth")
        scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
        le_path = os.path.join(artifacts_dir, "label_encoder.pkl")

        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path)):
            raise FileNotFoundError(
                f"Artifacts not found in '{artifacts_dir}'. "
                f"Expected model.pt, scaler.pkl, label_encoder.pkl"
            )

        ckpt = torch.load(model_path, map_location=torch.device("cpu"))

        self.model = IrisMLP(
            input_dim = ckpt["input_dim"],
            hidden_dim = ckpt["hidden_dim"],
            num_classes = ckpt["num_classes"],
        )
        self.model.load_state_dict(ckpt["state_dict"])

        self.model.eval()

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        with open(le_path, "rb") as f:
            self.label_encoder = pickle.load(f)

    def predict_one(self, features: dict) -> dict:
        # Validate & order features
        x = []
        for c in FEATURE_COLS:
            if c not in features:
                raise ValueError(f"Missing feature '{c}'. Required: {FEATURE_COLS}")
            x.append(features[c])

        x = np.array([x], dtype="float32")
        x_scaled = self.scaler.transform(x).astype("float32")

        with torch.no_grad():
            logits = self.model(torch.from_numpy(x_scaled))
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = str(self.label_encoder.inverse_transform([pred_idx])[0])

        return {
            "prediction": pred_label,
            "probabilities": {
                str(cls): float(probs[i]) for i, cls in enumerate(self.label_encoder.classes_)
            }
        }