"""
This example shows how to use grid search to find the minimum of a function. We will use the function (x - 2) ** 2 + (y + 1) ** 2 + 3.
"""

# %%
# Plot (x - 2) ** 2 + (y + 1) ** 2 + 3
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# %%
# Plot surface of (x - 2) ** 2 + (y + 1) ** 2 + 3
x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, y)
Z = (X - 2) ** 2 + (Y + 1) ** 2 + 3

plt.figure(1)
plt.contourf(X, Y, Z, levels=50, cmap="viridis")
plt.colorbar(label="Function Value")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contour Plot of (x - 2)^2 + (y + 1)^2 + 3")
plt.scatter(2, -1, color="red", marker="x", s=100, label="Minimum (2, -1)")
plt.legend()
# plt.show()

# %%
# Plot (x - 2) ** 2 + (y + 1) ** 2 + 3 in 3D
fig = plt.figure(2, figsize=(10.5, 7.5))
ax = fig.add_subplot(111, projection="3d")

# Plot the surface
ax.plot_surface(
    X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.8, rstride=5, cstride=5
)

# Add labels and title
ax.view_init(elev=45, azim=45)
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.set_zlabel("f(x, y)")
ax.set_title("3D Surface Plot of (x - 2)^2 + (y + 1)^2 + 3")

# Mark the minimum point
ax.scatter(2, -1, 3, color="red", s=100, label="Minimum (2, -1, 3)")
ax.legend()
fig.tight_layout()
plt.show()

# %%
# Use generated data for grid search
print(f"Using generated data for grid search")
# Find X, Y that minimize Z array
min_index = np.unravel_index(np.argmin(Z), Z.shape)
min_x = X[min_index]
min_y = Y[min_index]
print(f"Minimum value of Z is {Z[min_index]} at (x, y) = ({min_x}, {min_y})")
print(f"")

print("Number of function evaluations:", Z.size)
