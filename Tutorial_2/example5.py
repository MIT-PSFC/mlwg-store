"""
Langermann function (Optimization benchmark) visualization with noise addition and 3D surface plot using Plotly.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

SEED = 42
np.random.seed(SEED)


# Define the Langermann function
def langermann(x, y):
    # Langermann function parameters
    m = 5
    c = np.array([1, 2, 5, 2, 3])
    A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])

    result = 0
    for i in range(m):
        xi = x - A[i, 0]
        yi = y - A[i, 1]
        result += (
            c[i]
            * np.exp(-(1 / np.pi) * (xi**2 + yi**2))
            * np.cos(np.pi * (xi**2 + yi**2))
        )
    return result


# Create a grid of x and y values
x = np.linspace(0, 10, 1000)
y = np.linspace(0, 10, 1000)
X, Y = np.meshgrid(x, y)

# Noise addition to the function values
noise = np.random.normal(0, 0.25, X.shape)

# Compute the Langermann function values
Z = langermann(X, Y)  # + noise


# Plot the function
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=50, cmap="viridis")
plt.colorbar(label="Function Value")

plt.title("Langermann Function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# Create a 3D surface plot
fig = go.Figure(
    data=[
        go.Surface(
            z=Z,
            x=X[0],
            y=Y[:, 0],
            colorscale="Viridis",
        )
    ]
)

# Update layout for better visualization
fig.update_layout(
    title="Langermann Function (3D)",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="Function Value",
        aspectmode="cube",
    ),
    autosize=True,
)

# Show the plot
# fig.show()
fig.write_html("langermann_3d_surface.html", auto_open=True)
