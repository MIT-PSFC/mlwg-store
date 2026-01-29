import numpy as np
import matplotlib.pyplot as plt
import optuna

SEED = 42
np.random.seed(SEED)



# Define the Eggholder function
def eggholder(x, y):
    return -(y + 47.) * np.sin(np.sqrt(abs(x / 2. + (y + 47.)))) - x * np.sin(np.sqrt(abs(x - (y + 47.))))

# # Define the objective function for Optuna
def objective(trial):
    x = trial.suggest_float('x', -512, 512)
    y = trial.suggest_float('y', -512, 512)
    return eggholder(x, y)

# Create a study and optimize the objective function

# Sampler options:

# AutoSampler from optunahub
# import optunahub
# module = optunahub.load_module(package="samplers/auto_sampler")
# sampler = module.AutoSampler(seed=SEED)

# sampler = optuna.samplers.NSGAIISampler(seed=SEED)
# sampler = optuna.samplers.TPESampler(seed=SEED) 
# sampler = optuna.samplers.GPSampler(seed=SEED) # GP process works the best for this function

# Create a study and optimize the objective function

# Got lucky with Optuna 4.5.0
sampler = optuna.samplers.GPSampler(seed=SEED)
study = optuna.create_study(direction='minimize', 
                            sampler=sampler)
study.optimize(objective, n_trials= 600,  show_progress_bar=True)  

# Studies options with pruners (no really worth it sine function is cheap to evaluate)
# study = optuna.create_study(direction='minimize',
#                             sampler=sampler,
#                             pruner=optuna.pruners.MedianPruner( ))


# Another option
# sampler = optuna.samplers.GPSampler(seed=SEED)
# study = optuna.create_study(direction='minimize', sampler=sampler, 
#                             pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=1)              
#                             )
# study.optimize(objective, n_trials= 600, n_jobs=1, show_progress_bar=True) 

# Another option
# TPESampler needs more trials and it tends to stuck in local minima
# sampler = optuna.samplers.GPSampler(seed=SEED)
# study = optuna.create_study(direction='minimize', sampler=sampler)
# study.optimize(objective, n_trials= 600, n_jobs=1, show_progress_bar=True) 

# Another option
# TPESampler needs more trials and it tends to stuck in local minima
# sampler = optuna.samplers.TPESampler(seed=SEED)
# study = optuna.create_study(direction='minimize', 
#                             sampler=sampler, 
#                             pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=1)
#                             )
# study.optimize(objective, n_trials= 10000, n_jobs=1, show_progress_bar=True)

# Get the best parameters and value
best_params = study.best_params
best_value = study.best_value

print(f"Best parameters: x = {best_params['x']}, y = {best_params['y']}")
print(f"Benchmark solution: x* = 512, y* = 404.2319")
print(f"Minimum value: {best_value}")
print(f"Benchmark solution: f(x*,y*)= -959.6407")
print(f"Difference f(x,y) with global optimum: {abs(-959.6407 - best_value)}")
