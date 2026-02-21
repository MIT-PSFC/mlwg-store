from pathlib import Path
import numpy as np
import pandas as pd
import torch

import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

from mkgp.core.kernels import SE_Kernel
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
kernel = SE_Kernel(1.0e0, 1.0e-1)

# This is only necessary if using kernel restart option on the data fitting
kernel_hyppar_bounds = np.atleast_2d([[1.0e-1, 1.0e-1], [1.0e1, 1.0e0]])

# Define a kernel to fit the given y-errors, needed for rigourous estimation of fit error including data error
error_kernel = SE_Kernel(1.0e0, 1.0e-1)

# Again, this is only necessary if using kernel restart option on the error fitting
error_kernel_hyppar_bounds = np.atleast_2d([[1.0e-1, 1.0e-1], [1.0e1, 1.0e0]])


# GPR fit rigourously accounting only for y-errors (this is the recommended option)
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


gpr.GPRFit(fit_x, hsgp_flag=True)
fit_y, fit_ye, fit_dy, fit_dye = gpr.get_gp_results()
nonoise_fit_y, nonoise_fit_ye, nonoise_fit_dy, nonoise_fit_dye = gpr.get_gp_results(noise_flag=False)
num_samples = 10
fit_y_samples = gpr.sample_GP(num_samples, actual_noise=False)

plot_save_directory = Path('./mkgp_1d_output')
plot_save_directory.mkdir(parents=True, exist_ok=True)
plot_num_samples = num_samples if num_samples < 10 else 10
plot_sigma = 2.0

norm = Normalize(vmin=-plot_num_samples+1, vmax=plot_num_samples-1, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap='GnBu')

# Raw data with GPR fit and error, only accounting for y-errors
save_file1 = plot_save_directory / 'mkgp_1d_fit_test_sampling.png'
fig3 = plt.figure(1)
ax3 = fig3.add_subplot(111)
plot_train_ye = plot_sigma * train_ye
plot_fit_y_lower = fit_y - plot_sigma * fit_ye
plot_fit_y_upper = fit_y + plot_sigma * fit_ye
ax3.errorbar(train_x.flatten(), train_y.flatten(), yerr=plot_train_ye.flatten(), ls='', marker='.', color='k')
ax3.plot(fit_x.flatten(), fit_y.flatten(), color='r')
ax3.fill_between(fit_x.flatten(), plot_fit_y_lower.flatten(), plot_fit_y_upper.flatten(), facecolor='r', edgecolor='None', alpha=0.2)
for ii in range(plot_num_samples):
    pc = mapper.to_rgba([ii])
    ax3.plot(fit_x.flatten(), fit_y_samples[ii, ...].flatten(), lw=1, c=pc)
ax3.set_xlim(0.5, 1.1)
#fig3.savefig(save_file3)

# Derivative of GPR fit and error, only accounting for y-errors
save_file4 = plot_save_directory / 'mkgp_1d_derivative_test_sampling.png'
fig4 = plt.figure(2)
plot_fit_dy_lower = fit_dy - plot_sigma * fit_dye
plot_fit_dy_upper = fit_dy + plot_sigma * fit_dye
ax4 = fig4.add_subplot(111)
ax4.plot(fit_x.flatten(), fit_dy.flatten(), color='r')
ax4.fill_between(fit_x.flatten(), plot_fit_dy_lower.flatten(), plot_fit_dy_upper.flatten(), facecolor='r', edgecolor='None', alpha=0.2)
for ii in range(plot_num_samples):
    pc = mapper.to_rgba([ii])
    ax4.plot(fit_x.flatten(), np.gradient(fit_y_samples[ii, ...].flatten(), fit_x.flatten()), lw=1, c=pc)
ax4.set_xlim(0.5, 1.1)
#fig4.savefig(save_file4)

plt.show()
