"""
In this example we demonstrate simultaneous hyperparameter optimization and
model selection using Optuna on the Iris dataset.
"""

# %%
import sklearn
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import optuna
import numpy as np
 
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


# %%
SEED = 42
np.random.seed(SEED)


def objective(trial):

    classifier_name = trial.suggest_categorical(
        "classifier",
        [
            "SVC",
            "NaiveBayes",
        ],
    )

    scale = trial.suggest_categorical("scale", [True, False])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")

    elif classifier_name == "NaiveBayes":
        classifier_obj = sklearn.naive_bayes.GaussianNB()

    clf_pipeline = make_pipeline(StandardScaler() if scale else None, classifier_obj)

    score = cross_val_score(clf_pipeline, X_train, y_train, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


# %%
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
classifier_name = best_params.pop("classifier")
scale = best_params.pop("scale")

if classifier_name == "SVC":
    classifier_obj = sklearn.svm.SVC(C=best_params["svc_c"], gamma="auto")
elif classifier_name == "NaiveBayes":
    classifier_obj = sklearn.naive_bayes.GaussianNB()

clf_final = make_pipeline(StandardScaler() if scale else None, classifier_obj)

clf_pipeline = make_pipeline(StandardScaler() if scale else None, classifier_obj)
clf_pipeline.fit(X_train, y_train)
y_pred_final = clf_pipeline.predict(X_test)

# %%
print("Final model accuracy:", accuracy_score(y_test, y_pred_final))
print(classification_report(y_test, y_pred_final, target_names=data.target_names))
print("Final model confusion matrix:\n", confusion_matrix(y_test, y_pred_final))
# Accuracy should be higher than the baseline model!

# %%
