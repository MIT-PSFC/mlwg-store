from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


class ExampleGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


input_vars = ['r', 'a', 'ip', 'bt', 'kappa', 'delta', 'zeffped', 'betan', 'neped', 'nsfrac']
output_vars = ['ptop']
df = pd.read_hdf('sample_data_10d_surrogate.h5', key='/data')
train_data, test_data = train_test_split(df, test_size=100, shuffle=False)

train_x = train_data[input_vars].to_numpy()
train_y = train_data[output_vars].to_numpy().flatten()
test_x = test_data[input_vars].to_numpy()
test_y = test_data[output_vars].to_numpy().flatten()
tensor_train_x = torch.tensor(train_x)
tensor_train_y = torch.tensor(train_y)
tensor_test_x = torch.tensor(test_x)
tensor_test_y = torch.tensor(test_y)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExampleGPModel(tensor_train_x, tensor_train_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# Marginal log likelihood loss function
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Training loop
n_iterations = 500
for i in range(n_iterations):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(tensor_train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, tensor_train_y).sum()
    loss.backward()
    print('Iteration %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, n_iterations, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()

model.eval()
likelihood.eval()
tensor_check_y = likelihood(model(tensor_train_x))
tensor_fit_y = likelihood(model(tensor_test_x))
check_y = tensor_check_y.mean.detach().numpy()
check_ye = np.sqrt(tensor_check_y.variance.detach().numpy())
fit_y = tensor_fit_y.mean.detach().numpy()
fit_ye = np.sqrt(tensor_fit_y.variance.detach().numpy())
score = r2_score(fit_y, test_y, multioutput='raw_values')
print(f'R2 score: {score}')


plot_save_directory = Path('./gpytorch_10d_output')
plot_save_directory.mkdir(parents=True, exist_ok=True)
plot_sigma = 2.0

# Raw data with GPR fit and error, only accounting for y-errors
save_file1 = plot_save_directory / 'gpytorch_10d_surrogate_test.png'
fig1 = plt.figure(1, figsize=(8, 6))
ax1 = fig1.add_subplot(111)
ax1.errorbar(train_y.flatten(), check_y.flatten(), yerr=plot_sigma * check_ye.flatten(), ls='', marker='+', color='b', label='Training')
ax1.errorbar(test_y.flatten(), fit_y.flatten(), yerr=plot_sigma * fit_ye.flatten(), ls='', marker='s', color='r', label='Testing')
ax1.annotate(r'$R^2_{test} = ' + f'{score[0]:.6f}' + r'$', xy=(50.0, 780.0), xytext=(50.0, 780.0))
xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
lmin = np.nanmin([xmin, ymin])
lmax = np.nanmin([xmax, ymax])
ax1.plot([lmin, lmax], [lmin, lmax], ls='--', c='k')
ax1.set_xlim(lmin, lmax)
ax1.set_ylim(lmin, lmax)
ax1.legend(loc='best')
fig1.savefig(save_file1)


df_scan = pd.read_hdf('sample_test_10d_surrogate.h5', key='/data')
scan_x = df_scan[input_vars].to_numpy()
scan_y = df_scan[output_vars].to_numpy().flatten()
scan_var = 'neped'
scan_points = 101
scan_range = abs(df_scan[scan_var].max() - df_scan[scan_var].min())
smin = float(df_scan[scan_var].min() - 0.05 * scan_range)
smax = float(df_scan[scan_var].max() + 0.05 * scan_range)
scan = np.linspace(smin, smax, scan_points)
scan_data = [np.full((scan_points, ), float(df_scan[key].mean())) for key in input_vars if key in df_scan]
scan_data[input_vars.index(scan_var)] = scan.copy()
scan_fit_x = np.stack(scan_data, axis=-1)
tensor_scan_fit_x = torch.tensor(scan_fit_x)

tensor_scan_fit_y = likelihood(model(tensor_scan_fit_x))
scan_fit_y = tensor_scan_fit_y.mean.detach().numpy()
scan_fit_ye = np.sqrt(tensor_scan_fit_y.variance.detach().numpy())

# Raw data with GPR fit and error, only accounting for y-errors
save_file2 = plot_save_directory / 'gpytorch_10d_surrogate_test_scan.png'
fig2 = plt.figure(2, figsize=(8, 6))
ax2 = fig2.add_subplot(111)
plot_fit_y_lower = scan_fit_y - plot_sigma * scan_fit_ye
plot_fit_y_upper = scan_fit_y + plot_sigma * scan_fit_ye
ax2.scatter(df_scan[scan_var].to_numpy().flatten(), scan_y.flatten(), ls='', marker='o', color='k', label='Truth')
ax2.plot(scan.flatten(), scan_fit_y.flatten(), color='r', label='Surrogate')
ax2.fill_between(scan.flatten(), plot_fit_y_lower, plot_fit_y_upper, facecolor='r', edgecolor='None', alpha=0.2)
ax2.set_xlim(smin, smax)
ax2.legend(loc='best')
#fig2.savefig(save_file2)
plt.close(fig2)


plt.show()
