import numpy as np
import h5py
import matplotlib.pyplot as plt

# -----------------------
# Parameters
# -----------------------
seed = 42
n_per_cluster = 500
mean1 = np.array([0.0, 0.0])
mean2 = np.array([1.0, 1.0])
cov = np.array([[0.3, 0.0],
                [0.0, 0.3]])  # shared covariance
noise_std = 1.0

out_h5 = "data.h5"
out_fig = "gaussians_3d.png"

# -----------------------
# Generate data
# -----------------------
rng = np.random.RandomState(seed)

x1 = rng.multivariate_normal(mean=mean1, cov=cov, size=n_per_cluster)
x2 = rng.multivariate_normal(mean=mean2, cov=cov, size=n_per_cluster)

X2 = np.vstack([x1, x2])  # (N,2)
labels = np.hstack([
    np.zeros(n_per_cluster, dtype=int),
    np.ones(n_per_cluster, dtype=int)
])  # (N,)

# Add 1D pure noise as the 3rd feature
noise = rng.normal(loc=0.0, scale=noise_std, size=(X2.shape[0], 1))
X_obs = np.hstack([X2, noise])  # (N,3) = [x, y, noise]

# -----------------------
# Save to HDF5
# -----------------------
with h5py.File(out_h5, "w") as f:
    f.create_dataset("X_obs", data=X_obs)   # (N,3)
    f.create_dataset("labels", data=labels) # (N,)

    meta = f.create_group("meta")
    meta.attrs["description"] = "Two 2D Gaussian clusters plus one pure-noise dimension."
    meta.attrs["n_per_cluster"] = int(n_per_cluster)
    meta.attrs["mean1"] = mean1
    meta.attrs["mean2"] = mean2
    meta.attrs["cov_row_major"] = cov.flatten()
    meta.attrs["noise_std"] = float(noise_std)
    meta.attrs["seed"] = int(seed)
print(f"Saved dataset to: {out_h5}")

# -----------------------
# 3D scatter (colored by cluster)
# -----------------------
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(X_obs[:, 0], X_obs[:, 1], X_obs[:, 2], c=labels, s=10)
ax.set_title("Two 2D Gaussians + 1D Noise (colored by cluster)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("noise")

plt.tight_layout()
plt.savefig(out_fig, dpi=300)
plt.show()

# -----------------------
# 3D scatter (no color)
# -----------------------
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_obs[:, 0], X_obs[:, 1], X_obs[:, 2], color="gray", s=10)
ax.set_title("Two 2D Gaussians + 1D Noise (no color)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("noise")

plt.tight_layout()
plt.savefig(out_fig.replace(".png", "_gray.png"), dpi=300)
plt.show()
print(f"Saved grayscale 3D scatter to: {out_fig.replace('.png', '_gray.png')}")
