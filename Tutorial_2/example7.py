"""
In this example we demonstrate hyperparameter optimization of a neural network classifier on the Iris dataset using Optuna.
We will optimize hyperparameters such as the number of layers, number of units per layer, activation function, solver, learning rate, and batch size.
"""

# %%
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna
import numpy as np
from sklearn.model_selection import cross_val_score

# %%
### Load dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df["target_name"] = pd.Categorical.from_codes(data.target, data.target_names)

df.head()

# %%


X = df.drop(columns=["target", "target_name"])
y = df["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Train a baseline model
clf = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
        early_stopping=True,
    ),
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=data.target_names))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# %%
### Optuna hyperparameter optimization

SEED = 42
np.random.seed(SEED)


def objective(trial):
    scale = trial.suggest_categorical("scale", [True, False])
    n_layers = trial.suggest_int("n_layers", 1, 10)
    hidden = tuple(
        trial.suggest_int(f"n_units_l{i}", 4, 128, log=True) for i in range(n_layers)
    )
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
    solver = trial.suggest_categorical("solver", ["adam", "sgd"])
    alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    clf_trial = make_pipeline(
        StandardScaler() if scale else None,
        MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=2000,
            early_stopping=True,
            random_state=SEED,
        ),
    )

    score = np.mean(
        cross_val_score(clf_trial, X_train, y_train, cv=3, scoring="accuracy")
    )
    return score


sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=500)

# %%
print("Best trial:")
trial = study.best_trial

print("  Value: {:.4f}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %%
# Use the best hyperparameters to train a final model
best_params = trial.params
scale = best_params.pop("scale")
n_layers = best_params.pop("n_layers")
hidden = tuple(best_params.pop(f"n_units_l{i}") for i in range(n_layers))
clf_final = make_pipeline(
    StandardScaler() if scale else None,
    MLPClassifier(
        hidden_layer_sizes=hidden,
        **best_params,
        max_iter=2000,
        early_stopping=True,
        random_state=SEED,
    ),
)
clf_final.fit(X_train, y_train)
y_pred_final = clf_final.predict(X_test)

print("Final model accuracy:", accuracy_score(y_test, y_pred_final))
print(classification_report(y_test, y_pred_final, target_names=data.target_names))
print("Final model confusion matrix:\n", confusion_matrix(y_test, y_pred_final))
# Accuracy should be higher than the baseline model!
# %%
