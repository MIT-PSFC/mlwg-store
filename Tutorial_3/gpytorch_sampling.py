from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gpytorch

import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import Normalize
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


# Setup input data
input_vars = ['x']
output_vars = ['y']
output_error_vars = ['yerr']
df = pd.read_hdf('sample_data_1d_curvefit.h5', key='/data')

train_x = df[input_vars].to_numpy()
train_y = df[output_vars].to_numpy().flatten()
train_ye = df[output_error_vars].to_numpy().flatten()
tensor_train_x = torch.tensor(train_x)
tensor_train_y = torch.tensor(train_y)
tensor_train_ye = torch.tensor(train_ye)

# Define vector to evaluate fit
fit_x = np.linspace(0.5, 1.1, 101)
tensor_fit_x = torch.autograd.Variable(torch.tensor(fit_x), requires_grad=True)

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
n_iterations = 100
for i in range(n_iterations):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(tensor_train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, tensor_train_y).sum()
    loss.backward()
    optimizer.step()

model.eval()
likelihood.eval()


fit_posterior = likelihood(model(tensor_fit_x))
fit_y = fit_posterior.mean.detach().numpy()
fit_ye = np.sqrt(fit_posterior.variance.detach().numpy())
#fit_dy = torch.autograd.grad(fit_posterior.mean.sum(), tensor_fit_x)[0].detach().numpy()
#fit_dye = np.sqrt(torch.autograd.grad(fit_posterior.variance.sum(), tensor_fit_x)[0].detach().numpy())
fit_dy = torch.autograd.functional.jacobian(lambda x: likelihood(model(x)).mean.sum(), tensor_fit_x)
fit_dye = torch.autograd.functional.jacobian(lambda x: likelihood(model(x)).variance.sum(), tensor_fit_x)
num_samples = 10
fit_y_samples = fit_posterior.rsample(sample_shape=torch.Size([num_samples]))

plot_save_directory = Path('./gpytorch_1d_output')
plot_save_directory.mkdir(parents=True, exist_ok=True)
plot_num_samples = num_samples if num_samples < 10 else 10
plot_sigma = 2.0

norm = Normalize(vmin=-plot_num_samples+1, vmax=plot_num_samples-1, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap='GnBu')

# Raw data with GPR fit and error, only accounting for y-errors
save_file3 = plot_save_directory / 'gpytorch_1d_fit_test_sampling.png'
fig3 = plt.figure(1)
ax3 = fig3.add_subplot(111)
plot_train_ye = plot_sigma * train_ye
plot_fit_y_lower = fit_y - plot_sigma * fit_ye
plot_fit_y_upper = fit_y + plot_sigma * fit_ye
ax3.errorbar(train_x, train_y, yerr=plot_train_ye, ls='', marker='.', color='k')
ax3.plot(fit_x, fit_y, color='r')
ax3.fill_between(fit_x, plot_fit_y_lower, plot_fit_y_upper, facecolor='r', edgecolor='None', alpha=0.2)
for ii in range(plot_num_samples):
    pc = mapper.to_rgba([ii])
    ax3.plot(fit_x, fit_y_samples[ii].detach().numpy(), lw=1, c=pc)
ax3.set_xlim(0.5, 1.1)
fig3.savefig(save_file3)

# Derivative of GPR fit and error, only accounting for y-errors
save_file4 = plot_save_directory / 'gpytorch_1d_derivative_test_sampling.png'
fig4 = plt.figure(2)
ax4 = fig4.add_subplot(111)
plot_fit_dy_lower = fit_dy - plot_sigma * fit_dye
plot_fit_dy_upper = fit_dy + plot_sigma * fit_dye
ax4.plot(fit_x, fit_dy, color='r')
ax4.fill_between(fit_x, plot_fit_dy_lower, plot_fit_dy_upper, facecolor='r', edgecolor='None', alpha=0.2)
for ii in range(plot_num_samples):
    pc = mapper.to_rgba([ii])
    ax4.plot(fit_x, np.gradient(fit_y_samples[ii].detach().numpy(), fit_x), lw=1, c=pc)
ax4.set_xlim(0.5, 1.1)
fig4.savefig(save_file4)

plt.show()
