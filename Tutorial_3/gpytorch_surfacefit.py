from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gpytorch
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


# Setup input data
input_vars = ['t', 'x']
output_vars = ['y']
output_error_vars = ['yerr']
df = pd.read_hdf('sample_data_2d_surfacefit.h5', key='/data')

train_x = df[input_vars].to_numpy()
train_y = df[output_vars].to_numpy().flatten()
train_ye = df[output_error_vars].to_numpy().flatten()
tensor_train_x = torch.tensor(train_x)
tensor_train_y = torch.tensor(train_y)
tensor_train_ye = torch.tensor(train_ye)

# Define vector to evaluate fit
mesh_1, mesh_2 = np.meshgrid(np.linspace(0.0, 1.0, 31), np.linspace(0.5, 1.1, 51))
fit_x = np.stack([mesh_1.flatten(), mesh_2.flatten()], axis=1)
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
    print('Iteration %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, n_iterations, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()

model.eval()
likelihood.eval()
score_y = likelihood(model(tensor_train_x)).mean.detach().numpy()
score = r2_score(score_y, train_y, multioutput='raw_values')
print(f'R2 score: {score}')


fit_posterior = likelihood(model(tensor_fit_x))
fit_y = fit_posterior.mean.detach().numpy()
fit_ye = np.sqrt(fit_posterior.variance.detach().numpy())
fit_dy = torch.autograd.functional.jacobian(lambda x: likelihood(model(x)).mean.sum(), tensor_fit_x)
fit_dye = torch.autograd.functional.jacobian(lambda x: likelihood(model(x)).variance.sum(), tensor_fit_x)

plot_save_directory = Path('./gpytorch_2d_output')
plot_save_directory.mkdir(parents=True, exist_ok=True)
plot_sigma = 2.0

# Raw data with GPR fit and error, only accounting for y-errors
save_file1 = plot_save_directory / 'gpytorch_2d_fit_test.png'
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

# Derivative of GPR fit and error, only accounting for y-errors
save_file2 = plot_save_directory / 'gpytorch_2d_derivative_test.png'
fig2 = plt.figure(2)
ax2_1 = fig2.add_subplot(121, projection='3d')
ax2_2 = fig2.add_subplot(122, projection='3d')
plot_fit_dy_lower = fit_dy - plot_sigma * fit_dye
plot_fit_dy_upper = fit_dy + plot_sigma * fit_dye
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

plt.show()
