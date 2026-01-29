"""
Visualization of the Eggholder function, a complex optimization benchmark function, using contour and 3D surface plots.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

SEED = 42
np.random.seed(SEED)


# %%
# Define the Eggholder function
def eggholder(x, y):
    return -(y + 47) * np.sin(np.sqrt(abs(x / 2 + (y + 47)))) - x * np.sin(
        np.sqrt(abs(x - (y + 47)))
    )


# Create a grid of x and y values
x = np.linspace(-512, 512, 1000)
y = np.linspace(-512, 512, 1000)
X, Y = np.meshgrid(x, y)

# Compute the Eggholder function values
Z = eggholder(X, Y)

# Add marker for the global minimum
min_x, min_y = 512, 404.2319
min_z = eggholder(min_x, min_y)
print(f"Global minimum at x={min_x}, y={min_y}, z={min_z}")

# Plot the function
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=50, cmap="viridis")
plt.colorbar(label="Function Value")
plt.plot(min_x, min_y, "ro")  # Red dot at the minimum
plt.text(min_x - 220, min_y - 5, "Global Minimum", color="red")

plt.text(min_x - 100, min_y - 45, "-959.6407", color="red")

plt.title("Eggholder Function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %%
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
    title="Eggholder Function (3D)",
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
fig.write_html("eggholder_3d_surface.html", auto_open=True)
