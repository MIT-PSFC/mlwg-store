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


input_vars = ['r', 'a', 'ip', 'bt', 'kappa', 'delta', 'zeffped', 'betan', 'neped']
output_vars = ['ptop']
df = pd.read_hdf('sample_data_9d_surrogate.h5', key='/data')
train_data, test_data = train_test_split(df, test_size=10, shuffle=False)

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
optimizer = torch.optim.Adam(model.parameters(), lr=1.0)  # Includes GaussianLikelihood parameters

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
score = r2_score(likelihood(model(tensor_test_x)).mean.detach().numpy(), test_y, multioutput='raw_values')
print(f'R2 score: {score}')
