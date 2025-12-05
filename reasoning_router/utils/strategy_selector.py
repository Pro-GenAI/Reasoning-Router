import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from reasoning_router.utils.llm_utils import encode


class StrategyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


MODEL_DIR = "models/strategy_classifier"


class StrategyClassifier:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.label_to_strategy = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        model_path = os.path.join(self.model_dir, "classifier.pth")
        config_path = os.path.join(self.model_dir, "config.json")
        labels_path = os.path.join(self.model_dir, "labels.json")

        if (
            os.path.exists(model_path)
            and os.path.exists(config_path)
            and os.path.exists(labels_path)
        ):
            with open(config_path, "r") as f:
                config = json.load(f)
            self.model = StrategyNet(**config)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

            with open(labels_path, "r") as f:
                strategy_to_label = json.load(f)
                self.label_to_strategy = {v: k for k, v in strategy_to_label.items()}
        else:
            print("Model not found, using default strategy 'chain_of_thought'")

    def predict(self, text):
        if self.model is None:
            return "chain_of_thought"
        emb = encode(text)
        emb_tensor = torch.tensor(emb, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(emb_tensor)
            _, pred_label = torch.max(outputs, 1)
        return self.label_to_strategy[pred_label.item()]  # type: ignore


classifier = StrategyClassifier()
