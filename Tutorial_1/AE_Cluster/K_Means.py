import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -----------------------
# Configuration
# -----------------------
in_h5 = "data.h5"
random_seed = 42
k = 2
# -----------------------
# Load data
# -----------------------
with h5py.File(in_h5, "r") as f:
    X = f["X_obs"][:]          # (N, 3) = [x, y, noise]
    labels_gt = f["labels"][:] # ground-truth labels
# -----------------------
# K-means in 3D
# -----------------------
km = KMeans(n_clusters=k, n_init=10, random_state=random_seed)
pred = km.fit_predict(X)
centroids = km.cluster_centers_

# -----------------------
# Real accuracy via optimal label mapping (Hungarian)
# -----------------------
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

cm = confusion_matrix(labels_gt, pred, labels=np.unique(labels_gt))
row_ind, col_ind = linear_sum_assignment(-cm)  # maximize matches
acc = cm[row_ind, col_ind].sum() / cm.sum()
print(f"[KMeans 3D] Real accuracy vs saved labels = {acc:.4f}")
print("Centroids:\n", centroids)

# -----------------------
# Below Just Plotting and Decoration
# -----------------------
# -----------------------
# Smooth decision plane (k=2): (x - m)·n = 0 with n = c2 - c1, m = (c1 + c2)/2
# Render with a 2D grid (grid_res x grid_res) over two axes; solve the third.
# -----------------------
out_fig = "kmeans3d_with_decision_plane.png"
grid_res = 50  # resolution per axis for rendering the plane mesh
c1, c2 = centroids[0], centroids[1]
n = c2 - c1
m = 0.5 * (c1 + c2)
# Data bounds with padding
pad = 0.2
mins = X.min(axis=0) - pad
maxs = X.max(axis=0) + pad
# Choose the axis to solve for (largest |n_i| to avoid division issues)
abs_n = np.abs(n)
solve_axis = np.argmax(abs_n)
axes = [0, 1, 2]
free_axes = [a for a in axes if a != solve_axis]
u = np.linspace(mins[free_axes[0]], maxs[free_axes[0]], grid_res)
v = np.linspace(mins[free_axes[1]], maxs[free_axes[1]], grid_res)
UU, VV = np.meshgrid(u, v, indexing="xy")
# Prepare mesh placeholders
XX = np.zeros_like(UU)
YY = np.zeros_like(UU)
ZZ = np.zeros_like(UU)

# Assign the two known axes from the grid
XX[:] = UU if free_axes[0] == 0 else VV if free_axes[1] == 0 else XX
YY[:] = UU if free_axes[0] == 1 else VV if free_axes[1] == 1 else YY
ZZ[:] = UU if free_axes[0] == 2 else VV if free_axes[1] == 2 else ZZ

# Solve (X - m)·n = 0 for the remaining axis
NX, NY, NZ = n
MX, MY, MZ = m
eps = 1e-12

if solve_axis == 2:  # solve for z
    denom = NZ if np.abs(NZ) > eps else np.sign(NZ) * eps
    ZZ = MZ - (NX * (XX - MX) + NY * (YY - MY)) / denom
elif solve_axis == 1:  # solve for y
    denom = NY if np.abs(NY) > eps else np.sign(NY) * eps
    YY = MY - (NX * (XX - MX) + NZ * (ZZ - MZ)) / denom
else:  # solve for x
    denom = NX if np.abs(NX) > eps else np.sign(NX) * eps
    XX = MX - (NY * (YY - MY) + NZ * (ZZ - MZ)) / denom

# -----------------------
# Single figure: data + smooth decision plane; title shows real accuracy
# -----------------------
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

# Data (semi-transparent), colored by predicted cluster
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=pred, s=8, alpha=0.45)

# Smooth plane
ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, alpha=0.35, edgecolor='none')

# Centroids
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
           marker="X", s=180, c="red", edgecolor="white", linewidth=1.2, alpha=1.0, label="centroids")

ax.set_title(f"K-means decision plane (k=2) | Accuracy={acc:.4f}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("noise")
ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig(out_fig, dpi=300)
plt.show()
print(f"Saved 3D figure to: {out_fig}")
