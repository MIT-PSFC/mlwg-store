from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from mkgp.core.kernels import RQ_Kernel
from mkgp.core.routines import GaussianProcess

# Setup input data
input_vars = ['x']
output_vars = ['y']
output_error_vars = ['yerr']
df = pd.read_hdf('sample_data_1d_curvefit.h5', key='/data')

train_x = df[input_vars].to_numpy()
train_y = df[output_vars].to_numpy()
train_ye = df[output_error_vars].to_numpy()

# Define vector to evaluate fit
fit_x = np.atleast_2d(np.linspace(0.5, 1.1, 101)).T

# Define a kernel to fit the data itself
#     Rational quadratic kernel is usually robust enough for general fitting
kernel = RQ_Kernel(1.0e-1, 1.0e0, 5.0e0)

# This is only necessary if using kernel restart option on the data fitting
kernel_hyppar_bounds = np.atleast_2d([[1.0e-1, 1.0e-1, 1.0e0], [1.0e1, 1.0e0, 1.0e1]])

# Define a kernel to fit the given y-errors, needed for rigourous estimation of fit error including data error
#     Typically a simple rational quadratic kernel is sufficient given a high regularization parameter (specified later)
#     Here, the RQ kernel is summed with a noise kernel for extra robustness and to demonstrate how to use operator kernels
error_kernel = RQ_Kernel(1.0e-1, 1.0e0, 5.0e0)

# Again, this is only necessary if using kernel restart option on the error fitting
error_kernel_hyppar_bounds = np.atleast_2d([[1.0e-1, 1.0e-1, 1.0e0], [1.0e1, 1.0e0, 1.0e1]])


# GPR fit rigourously accounting only for y-errors (this is the recommended option)
#     Procedure is nearly identical to above, except for the addition of an error kernel
gpr = GaussianProcess()

#     Define the kernel and regularization parameter to be used in the data fitting routine
gpr.set_kernel(
    kernel=kernel,
    kbounds=kernel_hyppar_bounds,
    regpar=1.0
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
gpr.set_search_parameters(epsilon=1.0e-2, method='adam', spars=[1.0e-1, 0.9, 0.999])
gpr.set_error_search_parameters(epsilon=1.0e-1, method='adam', spars=[1.0e-1, 0.9, 0.999])

#     Perform the fit with kernel restarts
gpr.GPRFit(train_x, hsgp_flag=True, nrestarts=3)
fit_lml = gpr.get_gp_lml()
score_y, score_yv, score_dy, score_dyv = gpr.get_gp_results(rtn_cov=True)
from mkgp.core.utils import diagonalize
score = r2_score(score_y, train_y, multioutput='raw_values')
print(f'R2 score: {score}')


gpr.GPRFit(fit_x, hsgp_flag=True)
fit_y, fit_ye, fit_dy, fit_dye = gpr.get_gp_results()
nonoise_fit_y, nonoise_fit_ye, nonoise_fit_dy, nonoise_fit_dye = gpr.get_gp_results(noise_flag=False)

plot_save_directory = Path('./mkgp_1d_output')
plot_save_directory.mkdir(parents=True, exist_ok=True)
plot_num_samples = 10
plot_sigma = 2.0

# Raw data with GPR fit and error, only accounting for y-errors
save_file1 = plot_save_directory / 'mkgp_1d_fit_test.png'
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
plot_train_ye = plot_sigma * train_ye
plot_fit_y_lower = fit_y - plot_sigma * fit_ye
plot_fit_y_upper = fit_y + plot_sigma * fit_ye
ax1.errorbar(train_x.flatten(), train_y.flatten(), yerr=plot_train_ye.flatten(), ls='', marker='.', color='k')
ax1.plot(fit_x.flatten(), fit_y.flatten(), color='r')
ax1.fill_between(fit_x.flatten(), plot_fit_y_lower.flatten(), plot_fit_y_upper.flatten(), facecolor='r', edgecolor='None', alpha=0.2)
ax1.set_xlim(0.5, 1.1)
fig1.savefig(save_file1)
#plt.close(fig1)

# Derivative of GPR fit and error, only accounting for y-errors
save_file2 = plot_save_directory / 'mkgp_1d_derivative_test.png'
fig2 = plt.figure(2)
plot_fit_dy_lower = fit_dy - plot_sigma * fit_dye
plot_fit_dy_upper = fit_dy + plot_sigma * fit_dye
ax2 = fig2.add_subplot(111)
ax2.plot(fit_x.flatten(), fit_dy.flatten(), color='r')
ax2.fill_between(fit_x.flatten(), plot_fit_dy_lower.flatten(), plot_fit_dy_upper.flatten(), facecolor='r', edgecolor='None', alpha=0.2)
ax2.set_xlim(0.5, 1.1)
fig2.savefig(save_file2)
#plt.close(fig2)

plt.show()
