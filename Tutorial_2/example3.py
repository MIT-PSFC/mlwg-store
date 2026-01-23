"""
This example shows how to use Optuna to find the minimum of a function. We will use the function (x - 2) ** 2 + (y + 1) ** 2 + 3.
"""

# %%
import matplotlib.pyplot as plt
from prompt_toolkit import HTML
import numpy as np
import optuna

SEED = 42
np.random.seed(SEED)


# Define the function to optimize
def function_to_optimize(x, y):
    # Quadratic function: f(x,y) = (x-2)^2 + (y+1)^2 + 3
    # Minimum is at (2, -1) with value 3
    return (x - 2) ** 2 + (y + 1) ** 2 + 3


# Define the objective function for Optuna
def objective(trial):
    # Define parameters to optimize
    x = trial.suggest_float("x", -10.0, 10.0)
    y = trial.suggest_float("y", -10.0, 10.0)
    return function_to_optimize(x, y)


# Create study object and optimize
study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED)
)
study.optimize(objective, n_trials=100)

# Print results
print(f"Best parameters: {study.best_params}")
print(f"Best value: {study.best_value}")
print(f"Number of trials: {len(study.trials)}")

# %%
# Save study in a pickle file
import pickle

with open("optuna_study.pkl", "wb") as f:
    pickle.dump(study, f)

# %%
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Plot the surface
x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, y)
Z = (X - 2) ** 2 + (Y + 1) ** 2 + 3
ax.plot_surface(
    X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.6, rstride=5, cstride=5
)

# get x_path, y_path, z_path from the optimization history
x_path = [trial.params["x"] for trial in study.trials]
y_path = [trial.params["y"] for trial in study.trials]
z_path = [trial.value for trial in study.trials]

# Plot GD trajectory on the surface
ax.plot(
    x_path,
    y_path,
    z_path,
    color="red",
    linestyle="dashed",
    marker="o",
    markersize=3,
    label="GD path",
    alpha=0.7,
)
ax.scatter(x_path[0], y_path[0], z_path[0], color="green", s=60, label="start")
ax.scatter(x_path[-1], y_path[-1], z_path[-1], color="blue", s=60, label="end")

# Labels and view
ax.view_init(elev=45, azim=45)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
ax.set_title("3D Surface Plot with Gradient Descent Path")
ax.legend()

# plt.show()

# %%
plt.figure(figsize=(10, 7))
plt.contourf(X, Y, Z, levels=50, cmap="viridis")
plt.colorbar(label="Function Value")

# Plot GD trajectory (projected onto the x-y plane)
plt.plot(
    x_path,
    y_path,
    color="red",
    linestyle="dashed",
    linewidth=1,
    marker="o",
    markersize=3,
    label="GD path",
    alpha=0.7,
)
plt.scatter(x_path[0], y_path[0], color="green", s=60, label="start")
plt.scatter(x_path[-1], y_path[-1], color="blue", s=60, label="end")

# Mark true minimum
plt.scatter(
    2,
    -1,
    color="white",
    facecolor="black",
    marker="x",
    s=100,
    label="Minimum (2, -1)",
    zorder=5,
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Contour Plot with Gradient Descent Trajectory")
plt.legend()
plt.show()

# %%
from optuna.visualization import plot_contour, plot_optimization_history

# If Notebook not showing plotly figures directly, so we save it as an HTML file
import plotly.io as pio

fig = plot_contour(study)
pio.write_html(fig, file="contour_plot.html", auto_open=True)


fig = plot_optimization_history(study)
pio.write_html(fig, file="optimization_history.html", auto_open=True)
# %% To display the saved HTML file
from IPython.display import HTML

# (Next code for notebooks) Read the HTML file and display it in the notebook
# with open("contour_plot.html", "r") as f:
#      html_content1 = f.read()

# HTML(html_content1)

# with open("optimization_history.html", "r") as f:
#      html_content2 = f.read()

# HTML(html_content2)
