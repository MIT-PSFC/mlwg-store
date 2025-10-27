import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch

from Helmholtz import (
    se_cholesky,
    sample_grf_from_cov,
    rescale_to_minus1_1,
    solve_fd_dirichlet_scipy_sparse,
)
from FNO1D import FNO1d

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _denorm(u_norm, u_min, u_max):
    return (0.5 * (u_norm + 1.0)) * (u_max - u_min) + u_min

if __name__ == "__main__":

    DATA_H5    = "data.h5"
    MODEL_DIR  = "Results"
    SAVED_MODEL = os.path.join(MODEL_DIR, "best_fno1d.pt")
    OUT_FIG    = os.path.join(MODEL_DIR, f"test.png")
    
    # Read grid/meta + normalization from training data
    with h5py.File(DATA_H5, "r") as f:
        N = int(f.attrs["N"])
        L = float(f.attrs["L"])
        k = float(f.attrs["k"])
        u_min = float(f.attrs["solution_min"])
        u_max = float(f.attrs["solution_max"])

    x = np.linspace(0.0, L, N, endpoint=True)
    x_ch = torch.linspace(-1.0, 1.0, N, dtype=torch.float32)

    # GRF sample (mean 0) with l = L_TEST, rescaled to [-1,1] like the dataset
    SEED   = 1029
    L_TEST = 0.85 
    rng = np.random.default_rng(SEED)
    Lc = se_cholesky(x, l=L_TEST)
    S  = sample_grf_from_cov(Lc, rng)
    S  = rescale_to_minus1_1(S).astype(np.float32)

    # Reference solution (Dirichlet BCs)
    u_ref = solve_fd_dirichlet_scipy_sparse(x, S.astype(np.float64), k).astype(np.float32)

    # Load FNO
    model = FNO1d(modes=64, width=64, in_channels=2, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(SAVED_MODEL))
    model.eval()

    # Predict (normalized) then denormalize
    X_in = torch.stack([torch.from_numpy(S), x_ch], dim=0).unsqueeze(0).to(DEVICE) 
    with torch.no_grad():
        y_norm = model(X_in).cpu().squeeze().numpy()
    u_pred = _denorm(y_norm, u_min, u_max).astype(np.float32)

    # Simple metrics
    rmse = np.sqrt(np.mean((u_pred - u_ref) ** 2))
    relL2 = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-12)
    print(f"RMSE={rmse:.6e} | RelL2={relL2:.6e}")


    # Plot
    plt.figure(figsize=(9, 7))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(x, S, lw=1.6)
    ax1.set_ylabel("S(x)  [-1,1]")
    ax1.set_title(f"GRF source")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(x, u_ref, label="Reference (FD)", lw=1.8)
    ax2.plot(x, u_pred, "--", label="FNO", lw=1.6)
    ax2.set_xlabel("x")
    ax2.set_ylabel("u(x)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title(f"Helmholtz Equation\nRMSE={rmse:.3e},  RelL2={relL2:.3e}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.show()
    print(f"Saved figure: {OUT_FIG}")
