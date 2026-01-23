# Bayesian Optimization tutorial with OPTUNA
Prepared by Enrique Zapata for the Machine Learning Working Group of PSFC - MIT

This code collection shows how to use Optuna for different hyperparameter optimization applications.

### Installation

```sh
uv venv .venv
source .venv/bin/activate
uv sync
```

If using vscode select uv interpreter as the python interpreter. You need Python and Jupyter extensions.

Optimize something!
```python
import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}
```

### References
Check the official web and the docs!

https://optuna.org/

https://optuna.readthedocs.io/en/stable/ 

After taking a look at the examples, feel free to try the practical session in advance. 
Use Optuna and scikit-learn to obtain a perfect score on IRIS dataset.

Review the slides before the session
[Introduction to BO with Optuna (PDF)](Introduction%20to%20BO%20with%20Optuna%20EZ.pdf)

and the scikit-learn documentation...

https://scikit-learn.org/stable/supervised_learning.html  