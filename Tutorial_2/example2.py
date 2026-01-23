"""
This example shows how to use gradient descent to find the minimum of a function. We will use the function (x - 2) ** 2 + (y + 1) ** 2 + 3.
"""

# %%
### Use gradient descent to find the minimum of the function (x - 2) ** 2 + (y + 1) ** 2 + 3
import numpy as np


def function_to_minimize(x, y):
    return (x - 2) ** 2 + (y + 1) ** 2 + 3


def gradient(x, y, h=1e-5):
    f_x1 = function_to_minimize(x + h, y)
    f_x2 = function_to_minimize(x - h, y)
    f_y1 = function_to_minimize(x, y + h)
    f_y2 = function_to_minimize(x, y - h)
    grad_x = (f_x1 - f_x2) / (2 * h)
    grad_y = (f_y1 - f_y2) / (2 * h)
    return grad_x, grad_y


# Analytical gradient for the function (x - 2) ** 2 + (y + 1) ** 2 + 3
# def gradient(x, y):
#     grad_x = 2 * (x - 2)
#     grad_y = 2 * (y + 1)
#     return grad_x, grad_y

# Start close to a discontinuity (distance ≈ integer from below)
# Here, distance slightly under 3
x_current, y_current = 2 + 2.999, -1  # distance ≈2.999

learning_rate = 0.1
max_iterations = 100

print(
    f"Start: ({x_current}, {y_current}), f = {function_to_minimize(x_current, y_current)}"
)

for i in range(max_iterations):
    grad_x, grad_y = gradient(x_current, y_current)
    print(f"Iter {i}: grad = ({grad_x:.2f}, {grad_y:.2f})")

    x_next = x_current - learning_rate * grad_x
    y_next = y_current - learning_rate * grad_y

    # No convergence check to see wild behavior
    x_current, y_current = x_next, y_next
    print(
        f"   -> ({x_current:.4f}, {y_current:.4f}), f = {function_to_minimize(x_current, y_current)}"
    )

# %%
# Plot the function and the path taken by gradient descent
import matplotlib.pyplot as plt
import numpy as np

plt.figure(1)
x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, y)
Z = (X - 2) ** 2 + (Y + 1) ** 2 + 3
plt.contourf(X, Y, Z, levels=50, cmap="viridis")
plt.colorbar(label="Function Value")

# Recompute a gradient-descent path and plot it
start_x, start_y = 2 + 2.999, -1  # same start as in the GD cell
x_curr, y_curr = start_x, start_y

x_path = [x_curr]
y_path = [y_curr]
z_path = [function_to_minimize(x_curr, y_curr)]

for _ in range(max_iterations):
    gx, gy = gradient(x_curr, y_curr)
    x_curr -= learning_rate * gx
    y_curr -= learning_rate * gy
    x_path.append(x_curr)
    y_path.append(y_curr)
    z_path.append(function_to_minimize(x_curr, y_curr))

# Plot GD trajectory (projected onto the x-y plane)
plt.plot(
    x_path, y_path, color="red", linewidth=1, marker="o", markersize=3, label="GD path"
)
plt.scatter(x_path[0], y_path[0], color="green", s=60, label="start")
plt.scatter(x_path[-1], y_path[-1], color="blue", s=60, label="end")

# Mark true minimum
plt.scatter(
    2, -1, color="red", facecolor="black", marker="x", s=100, label="Minimum (2, -1)"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Contour Plot with Gradient Descent Trajectory")
plt.legend()
# plt.show()

# %%
# Plot the function in 3D and the path taken by gradient descent
fig = plt.figure(2, figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Plot the surface

ax.plot_surface(
    X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.6, rstride=5, cstride=5
)


# Plot GD trajectory on the surface
ax.plot(x_path, y_path, z_path, color="red", marker="o", markersize=3, label="GD path")
ax.scatter(x_path[0], y_path[0], z_path[0], color="green", s=60, label="start")
ax.scatter(x_path[-1], y_path[-1], z_path[-1], color="blue", s=60, label="end")

# Labels and view
ax.view_init(elev=45, azim=45)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
ax.set_title("3D Surface Plot with Gradient Descent Path")
ax.legend()

plt.show()

# %%
# Solution
print(
    f"Gradient Descent Solution: (x, y) = ({x_path[-1]:.4f}, {y_path[-1]:.4f}), f = {z_path[-1]:.4f}"
)

# %%
