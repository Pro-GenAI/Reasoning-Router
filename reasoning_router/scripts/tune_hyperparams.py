import optuna

from train_strategy_classifier import (
    load_and_prepare_data,
    train_classifier,
    save_model,
)

# Global data
texts = None
labels = None


def objective(trial):
    global texts, labels
    if texts is None:
        texts, labels = load_and_prepare_data()

    # Suggest hyperparameters
    hyperparams = {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 5, 20),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512, 1024]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }

    # Train model
    model, accuracy = train_classifier(texts, labels, hyperparams)

    return accuracy


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # type: ignore

    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)

    # Train final model with best hyperparams
    global texts, labels
    if texts is None:
        texts, labels = load_and_prepare_data()
    best_hyperparams = study.best_params
    model, accuracy = train_classifier(texts, labels, best_hyperparams)
    save_model(model, best_hyperparams)
    print("Best model saved!")


if __name__ == "__main__":
    main()
