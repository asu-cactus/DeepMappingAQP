
from torch import nn
import torch
import pandas as pd

class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, device):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pi = nn.Sequential(
            nn.Linear(in_features, out_features), nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features )
        self.mu = nn.Linear(in_features, out_features )

        self.pi = self.pi.to(device)
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu

device = 'cpu'

unit2size = []
for n_mdn_layer_node in range(310, 1000, 10):
    reg = nn.Sequential(
        nn.Linear(1, n_mdn_layer_node),
        nn.Tanh(),
        nn.Linear(n_mdn_layer_node, n_mdn_layer_node),
        nn.Tanh(),
        nn.Dropout(0.1),
        MDN(n_mdn_layer_node, 20, device),
    )

    kde = nn.Sequential(
        nn.Linear(2, n_mdn_layer_node),  # self.dim_input
        nn.Tanh(),
        nn.Linear(n_mdn_layer_node, n_mdn_layer_node),
        nn.Tanh(),
        nn.Dropout(0.1),
        MDN(n_mdn_layer_node, 5, device),
    )

    size = sum([p.nelement() * p.element_size() for p in reg.parameters()])
    size += sum([p.nelement() * p.element_size() for p in kde.parameters()])

    # print(f"Model size: {size / 1024:.2f} KB")
    unit2size.append({"units": n_mdn_layer_node, "size": round(size / 1024, 2)})

# Save the results
df = pd.DataFrame(unit2size)
df.to_csv("unit2size.csv", index=False)

