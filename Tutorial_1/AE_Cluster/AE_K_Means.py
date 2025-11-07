import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random

# -----------------------
# Reproducibility
# -----------------------
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# -----------------------
# Config
# -----------------------
in_h5 = "data.h5"
k = 2
epochs = 150
batch_size = 256
lr = 1e-3

# -----------------------
# Load data
# -----------------------
with h5py.File(in_h5, "r") as f:
    X = f["X_obs"][:]          # (N,3) = [x,y,noise]
    y_true = f["labels"][:]    # ground-truth labels for accuracy
# Standardize inputs for AE
scaler = StandardScaler()
X_std = scaler.fit_transform(X).astype(np.float32)

# -----------------------
# Define AE 3->2->3
# -----------------------
class AE(nn.Module):
    def __init__(self, in_dim=3, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32),     nn.ReLU(inplace=True),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 64),         nn.ReLU(inplace=True),
            nn.Linear(64, in_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z

# -----------------------
# Train AE
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE(in_dim=3, latent_dim=2).to(device)

dl = DataLoader(TensorDataset(torch.from_numpy(X_std)), batch_size=batch_size, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

best_loss, best_state = float("inf"), None
model.train()

for ep in range(1, epochs + 1):
    epoch_loss = 0.0
    for xb, in dl:
        xb = xb.to(device)
        opt.zero_grad()
        xhat, _ = model(xb)
        loss = loss_fn(xhat, xb)
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * xb.size(0)

    epoch_loss /= len(dl.dataset)
    if epoch_loss < best_loss - 1e-7:
        best_loss = epoch_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if ep % 20 == 0:
        print(f"Epoch {ep:03d} | Loss: {epoch_loss:.6f}")

# -----------------------
# Encode -> Z (N,2)
# -----------------------
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    Z = model.encoder(torch.from_numpy(X_std).to(device)).cpu().numpy()

# -----------------------
# K-means in latent(2D)
# -----------------------
km = KMeans(n_clusters=k, n_init=10, random_state=seed)
labels_pred = km.fit_predict(Z)
centroids_latent = km.cluster_centers_

# -----------------------
# Label-based accuracy
# -----------------------
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
cm = confusion_matrix(y_true, labels_pred)
r, c = linear_sum_assignment(-cm)
acc = cm[r, c].sum() / cm.sum()
print(f"[AE+KMeans] Accuracy = {acc:.4f}")
# Align predicted labels for consistent coloring
mapping = dict(zip(c, r))
labels_aligned = np.vectorize(lambda x: mapping.get(x, x))(labels_pred)


# -----------------------
# Below Just Plotting and Decoration
# -----------------------
latent_steps = 300       # grid for 2D latent decision map
grid_steps_3d = 50       # 3D grid resolution for boundary sampling
margin_factor = 0.02     # tie threshold in latent as fraction of centroid spacing

fig_latent_decision = "AE_latent_boundary_2D.png"
fig_boundary_3d = "AE_kmeans_boundary_3D.png"
# -----------------------
# Plot 1: 2D latent decision boundary (smooth contour) + points
# -----------------------
# Build latent grid
z1_min, z1_max = Z[:, 0].min() - 0.5, Z[:, 0].max() + 0.5
z2_min, z2_max = Z[:, 1].min() - 0.5, Z[:, 1].max() + 0.5
z1 = np.linspace(z1_min, z1_max, latent_steps)
z2 = np.linspace(z2_min, z2_max, latent_steps)
ZZ1, ZZ2 = np.meshgrid(z1, z2, indexing="xy")
G_latent = np.stack([ZZ1.ravel(), ZZ2.ravel()], axis=1)

# Distances to latent centroids & assignments
dists_latent = np.linalg.norm(G_latent[:, None, :] - centroids_latent[None, :, :], axis=2)
assign = np.argmin(dists_latent, axis=1).reshape(ZZ1.shape)

# Decision boundary = where two smallest distances are equal
# For k=2, simple difference contour at level 0:
if k == 2:
    diff = (dists_latent[:, 0] - dists_latent[:, 1]).reshape(ZZ1.shape)
else:
    # For k>2, use (d2 - d1) contour at 0 (more general)
    sorted_d = np.sort(dists_latent, axis=1)
    diff = (sorted_d[:, 1] - sorted_d[:, 0]).reshape(ZZ1.shape)

plt.figure(figsize=(6, 5))
# soft regions
plt.contourf(ZZ1, ZZ2, assign, levels=np.arange(k+1)-0.5, alpha=0.2)
# smooth boundary line(s)
plt.contour(ZZ1, ZZ2, diff, levels=[0.0], linewidths=2)
# data points
plt.scatter(Z[:, 0], Z[:, 1], c=labels_aligned, s=10, alpha=0.9)
# centroids
plt.scatter(centroids_latent[:, 0], centroids_latent[:, 1],
            marker="X", s=160, c="k", edgecolors="white", linewidths=1.2, label="centroids")

plt.title(f"AE latent (2D) decision boundary | Acc={acc:.4f}")
plt.xlabel("z1"); plt.ylabel("z2")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(fig_latent_decision, dpi=300)
plt.show()
print(f"Saved: {fig_latent_decision}")

# -----------------------
# Plot 2: 3D decision boundary (approx via encoded 3D grid near-ties)
# -----------------------
pad = 0.2
mins = X.min(axis=0) - pad
maxs = X.max(axis=0) + pad

xs = np.linspace(mins[0], maxs[0], grid_steps_3d)
ys = np.linspace(mins[1], maxs[1], grid_steps_3d)
zs = np.linspace(mins[2], maxs[2], grid_steps_3d)
XXg, YYg, ZZg = np.meshgrid(xs, ys, zs, indexing="xy")
G3 = np.stack([XXg.ravel(), YYg.ravel(), ZZg.ravel()], axis=1)

# Standardize, encode, compute latent distances for grid
G3_std = scaler.transform(G3).astype(np.float32)
with torch.no_grad():
    Zg = model.encoder(torch.from_numpy(G3_std).to(device)).cpu().numpy()

dists_g = np.linalg.norm(Zg[:, None, :] - centroids_latent[None, :, :], axis=2)
sorted_dg = np.sort(dists_g, axis=1)
d1g, d2g = sorted_dg[:, 0], sorted_dg[:, 1]

# Tie threshold in latent based on centroid spacing
pairs = []
for i in range(k):
    for j in range(i+1, k):
        pairs.append(np.linalg.norm(centroids_latent[i] - centroids_latent[j]))
latent_mean_spacing = np.mean(pairs) if pairs else 1.0
tau = margin_factor * latent_mean_spacing

boundary_mask = (d2g - d1g) < tau
boundary_pts_3d = G3[boundary_mask]

# Plot
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

# Original data (faint), colored by aligned predicted labels
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels_aligned, s=6, alpha=0.35)

# Boundary points (approx surface)
if boundary_pts_3d.size > 0:
    ax.scatter(boundary_pts_3d[:, 0], boundary_pts_3d[:, 1], boundary_pts_3d[:, 2],
               c="k", s=2, alpha=0.6, label="decision boundary")

ax.set_title(f"AE+k-means decision boundary in 3D | Acc={acc:.4f}")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("noise")

plt.tight_layout()
plt.savefig(fig_boundary_3d, dpi=300)
plt.show()
print(f"Saved: {fig_boundary_3d}")
