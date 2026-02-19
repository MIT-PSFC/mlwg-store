from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from mkgp.core.kernels import RQ_Kernel, ND_Sum_Kernel
from mkgp.core.routines import GaussianProcess

# Setup input data
input_vars = ['t', 'x']
output_vars = ['y']
output_error_vars = ['yerr']
df = pd.read_hdf('sample_data_2d_surfacefit.h5', key='/data')

train_x = df[input_vars].to_numpy()
train_y = df[output_vars].to_numpy()
train_ye = df[output_error_vars].to_numpy()

# Define vector to evaluate fit
mesh_1, mesh_2 = np.meshgrid(np.linspace(0.0, 1.0, 31), np.linspace(0.5, 1.1, 51))
fit_x = np.stack([mesh_1.flatten(), mesh_2.flatten()], axis=1)

# Define a kernel to fit the data itself
#     Rational quadratic kernel is usually robust enough for general fitting
kernel_1 = RQ_Kernel(1.0e-1, 1.0e0, 5.0e0)
kernel_2 = RQ_Kernel(1.0e-1, 1.0e0, 5.0e0)
kernel = ND_Sum_Kernel(kernel_1, kernel_2)

# This is only necessary if using kernel restart option on the data fitting
kernel_hyppar_bounds = np.atleast_2d([[1.0e-1, 1.0e-1, 1.0e0, 1.0e-1, 1.0e-1, 1.0e0], [1.0e1, 1.0e0, 1.0e1, 1.0e1, 1.0e0, 1.0e1]])

# Define a kernel to fit the given y-errors, needed for rigourous estimation of fit error including data error
#     Typically a simple rational quadratic kernel is sufficient given a high regularization parameter (specified later)
#     Here, the RQ kernel is summed with a noise kernel for extra robustness and to demonstrate how to use operator kernels
error_kernel_1 = RQ_Kernel(1.0e-1, 1.0e0, 5.0e0)
error_kernel_2 = RQ_Kernel(1.0e-1, 1.0e0, 5.0e0)
error_kernel = ND_Sum_Kernel(error_kernel_1, error_kernel_2)

# Again, this is only necessary if using kernel restart option on the error fitting
error_kernel_hyppar_bounds = np.atleast_2d([[1.0e-1, 1.0e-1, 1.0e0, 1.0e-1, 1.0e-1, 1.0e0], [1.0e1, 1.0e0, 1.0e1, 1.0e1, 1.0e0, 1.0e1]])


# GPR fit rigourously accounting only for y-errors (this is the recommended option)
#     Procedure is nearly identical to above, except for the addition of an error kernel
gpr = GaussianProcess()

#     Define the kernel and regularization parameter to be used in the data fitting routine
gpr.set_kernel(
    kernel=kernel,
    kbounds=kernel_hyppar_bounds,
    regpar=2.0
)

#     Define the kernel and regularization parameter to be used in the error fitting routine
gpr.set_error_kernel(
    kernel=error_kernel,
    kbounds=error_kernel_hyppar_bounds,
    regpar=5.0
)

#     Define the raw data and associated errors to be fitted
gpr.set_raw_data(
    xdata=train_x,
    ydata=train_y,
    yerr=train_ye,
    xerr=None,
    #dxdata=np.array([0.0]),           # Example of applying derivative constraints
    #dydata=np.array([0.0]),
    #dyerr=np.array([0.0])
)

#     Define the search criteria for data fitting routine and error fitting routine
gpr.set_search_parameters(epsilon=1.0e-1, method='adam', spars=[1.0e-1, 0.5, 0.9])
gpr.set_error_search_parameters(epsilon=1.0e-1, method='adam', spars=[1.0e-1, 0.5, 0.9])

#     Perform the fit with kernel restarts
gpr.GPRFit(train_x, hsgp_flag=True, nrestarts=5)
fit_lml = gpr.get_gp_lml()
score_y, score_yv, score_dy, score_dyv = gpr.get_gp_results(rtn_cov=True)
score = r2_score(score_y, train_y, multioutput='raw_values')
print(f'R2 score: {score}')


gpr.GPRFit(fit_x, hsgp_flag=True)
fit_y, fit_ye, fit_dy, fit_dye = gpr.get_gp_results()
nonoise_fit_y, nonoise_fit_ye, nonoise_fit_dy, nonoise_fit_dye = gpr.get_gp_results(noise_flag=False)

plot_save_directory = Path('./mkgp_2d_output')
plot_save_directory.mkdir(parents=True, exist_ok=True)
plot_num_samples = 10
plot_sigma = 2.0

# Raw data with GPR fit and error, only accounting for y-errors
save_file1 = plot_save_directory / 'mkgp_2d_fit_test.png'
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
plot_train_ye = plot_sigma * train_ye
plot_fit_y_lower = fit_y - plot_sigma * fit_ye
plot_fit_y_upper = fit_y + plot_sigma * fit_ye
ax1.errorbar(train_x[:, 0].flatten(), train_x[:, 1].flatten(), train_y.flatten(), zerr=plot_train_ye.flatten(), ls='', marker='.', color='k')
ax1.plot_surface(fit_x[:, 0].reshape(mesh_1.shape), fit_x[:, 1].reshape(mesh_2.shape), fit_y.reshape(mesh_1.shape), color='r', alpha=0.5)
ax1.plot_surface(fit_x[:, 0].reshape(mesh_1.shape), fit_x[:, 1].reshape(mesh_2.shape), plot_fit_y_lower.reshape(mesh_1.shape), facecolor='r', edgecolor='None', alpha=0.2)
ax1.plot_surface(fit_x[:, 0].reshape(mesh_1.shape), fit_x[:, 1].reshape(mesh_2.shape), plot_fit_y_upper.reshape(mesh_1.shape), facecolor='r', edgecolor='None', alpha=0.2)
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.5, 1.1)
fig1.savefig(save_file1)
#plt.close(fig1)

# Derivative of GPR fit and error, only accounting for y-errors
save_file2 = plot_save_directory / 'mkgp_2d_derivative_test.png'
fig2 = plt.figure(2)
plot_fit_dy_lower = fit_dy - plot_sigma * fit_dye
plot_fit_dy_upper = fit_dy + plot_sigma * fit_dye
ax2_1 = fig2.add_subplot(121, projection='3d')
ax2_2 = fig2.add_subplot(122, projection='3d')
ax2_1.plot_surface(fit_x[:, 0].reshape(mesh_1.shape), fit_x[:, 1].reshape(mesh_2.shape), fit_dy[:, 0].reshape(mesh_1.shape), color='r', alpha=0.5)
ax2_1.plot_surface(fit_x[:, 0].reshape(mesh_1.shape), fit_x[:, 1].reshape(mesh_2.shape), plot_fit_dy_lower[:, 0].reshape(mesh_1.shape), facecolor='r', edgecolor='None', alpha=0.2)
ax2_1.plot_surface(fit_x[:, 0].reshape(mesh_1.shape), fit_x[:, 1].reshape(mesh_2.shape), plot_fit_dy_upper[:, 0].reshape(mesh_1.shape), facecolor='r', edgecolor='None', alpha=0.2)
ax2_2.plot_surface(fit_x[:, 0].reshape(mesh_1.shape), fit_x[:, 1].reshape(mesh_2.shape), fit_dy[:, 1].reshape(mesh_1.shape), color='r', alpha=0.5)
ax2_2.plot_surface(fit_x[:, 0].reshape(mesh_1.shape), fit_x[:, 1].reshape(mesh_2.shape), plot_fit_dy_lower[:, 1].reshape(mesh_1.shape), facecolor='r', edgecolor='None', alpha=0.2)
ax2_2.plot_surface(fit_x[:, 0].reshape(mesh_1.shape), fit_x[:, 1].reshape(mesh_2.shape), plot_fit_dy_upper[:, 1].reshape(mesh_1.shape), facecolor='r', edgecolor='None', alpha=0.2)
ax2_1.set_xlim(0.0, 1.0)
ax2_1.set_ylim(0.5, 1.1)
ax2_2.set_xlim(0.0, 1.0)
ax2_2.set_ylim(0.5, 1.1)
fig2.savefig(save_file2)
#plt.close(fig2)

plt.show()
