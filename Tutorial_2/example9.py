"""
In this example we demonstrate multiobjective optimization
for feature selection
"""

# %%
### Load dataset
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import optuna
from optuna import Trial
from sklearn.naive_bayes import GaussianNB

# %%
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df["target_name"] = pd.Categorical.from_codes(data.target, data.target_names)

df.head()

# %%


X = df.drop(columns=["target", "target_name"])
y = df["target"]

X = X.to_numpy()
y = y.to_numpy()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)


# %%
SEED = 42
np.random.seed(SEED)


def objective(trial: Trial):
    # Suggest feature subset
    feature_mask = []
    for i in range(X.shape[1]):
        feature_mask.append(trial.suggest_categorical(f"feature_{i}", [True, False]))
    X_selected = X_train[:, feature_mask]
    number_of_features = sum(feature_mask)
    if X_selected.shape[1] == 0:
        return 0.0, number_of_features

    # Train and evaluate model
    clf = GaussianNB()
    score = cross_val_score(clf, X_selected, y_train, cv=5, scoring="accuracy")
    return score.mean(), number_of_features


# Create study (single-objective maximizing the combined metric)
sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)

# Enqueue the first trial with ALL features selected
all_features_params = {f"feature_{i}": True for i in range(X.shape[1])}
study.enqueue_trial(all_features_params)

# Now optimize (this will run the enqueued trial first, then 99 more)
study.optimize(objective, n_trials=300)

# %%
# Get Pareto front trials
# trials in pareto front
pareto_trials = study.best_trials
print("Pareto front trials:")
for trial in pareto_trials:
    print(
        f"Trial #{trial.number}: Score={trial.values[0]:.4f}, Number of features={trial.values[1]}, Params={trial.params}"
    )
# %%
# Visualize Pareto front
fig = optuna.visualization.matplotlib.plot_pareto_front(
    study,
    target_names=["Score", "Number of features"],
)
# %%
# Print selected features from one of the Pareto front trials
trial_to_inspect = 102  # 91
print(f"Selected features in trial #{trial_to_inspect}:")
trial = study.trials[trial_to_inspect]
selected_features = [i for i in range(X.shape[1]) if trial.params[f"feature_{i}"]]
print(selected_features)

# get name of selected features
feature_names = df.drop(columns=["target", "target_name"]).columns
print("Selected feature names:")
for i in selected_features:
    print(feature_names[i])
# %%
