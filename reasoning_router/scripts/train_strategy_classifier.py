# Train a model to select reasoning strategy name based on input question.

import json
import os

from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from reasoning_router.utils.strategy_selector import StrategyNet, MODEL_DIR
from reasoning_router.utils.llm_utils import encode


STRATEGY_LABELS = {
    "chain_of_thought": 0,
    "tree_of_thought": 1,
    "debate": 2,
    "self_reflection": 3,
}

# Default hyperparameters
HYPERPARAMS = {
    "batch_size": 32,
    "learning_rate": 0.002,
    "num_epochs": 7,
    "hidden_dim": 1024,
    "dropout": 0.17,
}

DATASET_CONFIG = {
    "gsm8k": {
        "name": "gsm8k",
        "config": "main",
        "split": "train",
        "text_field": "question",
        "strategy": "chain_of_thought",
        "limit": 100,
    },
    "ai2_arc": {
        "name": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "train",
        "text_field": "question",
        "strategy": "tree_of_thought",
        "limit": 100,
    },
    "hellaswag": {
        "name": "Rowan/hellaswag",
        "split": "train",
        "text_field": "ctx",
        "strategy": "debate",
        "limit": 100,
    },
    "truthful_qa": {
        "name": "truthfulqa/truthful_qa",
        "config": "generation",
        "split": "validation",
        "text_field": "question",
        "strategy": "self_reflection",
        "limit": 50,
    },
    "IFEval": {
        "name": "google/IFEval",
        "split": "train",
        "text_field": "prompt",
        "strategy": "self_reflection",
        "limit": 50,
    },
}


def load_and_prepare_data():
    texts = []
    labels = []

    for dataset_key, config in DATASET_CONFIG.items():
        print(f"Loading {dataset_key}...")
        try:
            if "config" in config:
                dataset = load_dataset(
                    config["name"], config["config"], split=config["split"]
                )
            else:
                dataset = load_dataset(config["name"], split=config["split"])
            strategy = config["strategy"]
            label = STRATEGY_LABELS[strategy]
            text_field = config["text_field"]
            limit = config["limit"]

            count = 0
            for item in dataset:
                if count >= limit:
                    break
                text = item[text_field]  # type: ignore
                texts.append(text)
                labels.append(label)
                count += 1
        except Exception as e:
            print(f"Error loading {dataset_key}: {e}")

    return texts, labels


def train_classifier(texts, labels, hyperparams=HYPERPARAMS):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Get embeddings
    X_train_emb = encode(X_train)
    X_test_emb = encode(X_test)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_emb, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_emb, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=hyperparams["batch_size"], shuffle=False
    )

    # Model
    input_dim = X_train_emb.shape[1]
    model = StrategyNet(
        input_dim=input_dim,
        hidden_dim=hyperparams["hidden_dim"],
        num_classes=len(STRATEGY_LABELS),
        dropout=hyperparams["dropout"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # Train
    num_epochs = hyperparams["num_epochs"]
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    print(
        classification_report(
            all_targets, all_preds, target_names=list(STRATEGY_LABELS.keys())
        )
    )

    accuracy = accuracy_score(all_targets, all_preds)
    return model, accuracy


def save_model(model, hyperparams=HYPERPARAMS, model_dir=MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "classifier.pth"))

    # Save model config
    config = {
        "input_dim": model.fc1.in_features,
        "hidden_dim": hyperparams["hidden_dim"],
        "num_classes": model.fc3.out_features,
        "dropout": hyperparams["dropout"],
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)

    # Save label mapping
    with open(os.path.join(model_dir, "labels.json"), "w") as f:
        json.dump(STRATEGY_LABELS, f)

    # Save hyperparameters
    with open(os.path.join(model_dir, "hyperparams.json"), "w") as f:
        json.dump(hyperparams, f)


if __name__ == "__main__":
    texts, labels = load_and_prepare_data()
    print(f"Loaded {len(texts)} samples")

    model, accuracy = train_classifier(texts, labels)
    save_model(model)
    print("Model saved!")
    print(f"Final accuracy: {accuracy:.4f}")
