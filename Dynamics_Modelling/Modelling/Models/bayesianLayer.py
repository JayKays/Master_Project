from typing import List, Sequence
from unicodedata import decimal

import torch
from torch import nn as nn
import torch.nn.functional as F

from blitz.modules.linear_bayesian_layer import BayesianLinear
from blitz.modules.base_bayesian_module import BayesianModule
from torch._C import is_grad_enabled

import numpy as np


def log_gaussian_prob(x, mu, sigma, log_sigma=False, ensemble = True):
    if not log_sigma:
        element_wise_log_prob = -0.5*torch.Tensor([np.log(2*np.pi)]).to(mu.device) - torch.log(sigma) - 0.5*(x-mu)**2 / sigma**2
    else:
        element_wise_log_prob = -0.5*torch.Tensor([np.log(2*np.pi)]).to(mu.device) - F.softplus(sigma) - 0.5*(x-mu)**2 / F.softplus(sigma)**2
    

    if ensemble:
        return element_wise_log_prob.sum(dim=(1,2))
    else:
        return element_wise_log_prob.sum()


class GaussianEnsembleLinearLayer(nn.Module):
    def __init__(self,num_members, in_dim, out_dim, stddev_prior = .001, bias=True):
        super(GaussianEnsembleLinearLayer, self).__init__()
        self.num_members = num_members
        self.is_ensemble = num_members > 1
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stddev_prior = stddev_prior
        self.w_mu = nn.Parameter(torch.Tensor(self.num_members, self.in_dim, self.out_dim).normal_(0, stddev_prior))
        self.w_rho = nn.Parameter(torch.Tensor(self.num_members, self.in_dim, self.out_dim).normal_(0, stddev_prior))
        if bias:
            self.use_bias = True
            self.b_mu = nn.Parameter(torch.Tensor(self.num_members, 1, self.out_dim).normal_(0, stddev_prior))
            self.b_rho = nn.Parameter(torch.Tensor(self.num_members, 1 , self.out_dim).normal_(0, stddev_prior))
        else:
            self.use_bias = False

        self.q_w = 0.
        self.p_w = 0.

    def forward(self, x):

        device = self.w_mu.device
        w_stddev = F.softplus(self.w_rho)
        # w = self.w_mu + w_stddev * torch.Tensor(*self.w_mu.shape).to(device).normal_(0,self.stddev_prior)
        w = self.w_mu + w_stddev * torch.Tensor(*self.w_mu.shape).to(device).normal_(0,1)
        self.q_w = log_gaussian_prob(w, self.w_mu, self.w_rho, log_sigma=True, ensemble = self.is_ensemble)
        self.p_w = log_gaussian_prob(w, torch.zeros_like(self.w_mu, device=device), self.stddev_prior*torch.ones_like(w_stddev, device=device))

        xw = x.matmul(w)

        if self.use_bias:
            b_stddev = F.softplus(self.b_rho)
            # print(*self.b_mu.shape)
            # b = self.b_mu + b_stddev * torch.Tensor(*self.b_mu.shape).to(device).normal_(0,self.stddev_prior)
            b = self.b_mu + b_stddev * torch.Tensor(*self.b_mu.shape).to(device).normal_(0,1)
            self.q_w += log_gaussian_prob(b, self.b_mu, self.b_rho, log_sigma=True, ensemble = self.is_ensemble)
            self.p_w += log_gaussian_prob(b, torch.zeros_like(self.b_mu, device=device), self.stddev_prior*torch.ones_like(b_stddev, device=device))
            
            return xw + b
        else: 
            return xw

    def get_pw(self):
        return self.p_w

    def get_qw(self):
        return self.q_w
    
    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, in_size={self.in_dim}, "
            f"out_size={self.out_dim}, bias={self.use_bias}"
        )


class EnsembleLinearBayesian(BayesianModule):

    """Efficient linear layer for ensemble models."""

    def __init__(
        self, num_members: int, in_size: int, out_size: int, bias: bool = True, freeze: bool = False, device ="cpu" 
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.use_bias = bias
        self.freeze = freeze
        
        self.member_layers = [BayesianLinear(self.in_size, self.out_size, self.use_bias) for _ in range(self.num_members)]
        
        for i in range(self.num_members):
            self.add_module(f"bl{i}", self.member_layers[i])

        self.log_variational_posterior = torch.zeros(self.num_members, device=device, requires_grad=True)
        self.log_prior = torch.zeros(self.num_members, device=device, requires_grad=True)

        # self.log_variational_posterior = 0
        # self.log_prior = 0

        self.elite_models: List[int] = None
        self.use_only_elite = False


    def forward(self, x):
        # if x.ndim == 3: x = x.squeeze()

        if x.ndim == 2: x = x.unsqueeze(0).repeat(self.num_members,1,1)
        assert x.shape[0] == self.num_members

        if self.freeze:
            xw = torch.stack([self.member_layers[i].forward_frozen(x[i]) for i in range(self.num_members)])
        else:    
            xw = torch.stack([self.member_layers[i].forward(x[i]) for i in range(self.num_members)])

        
        for i in range(self.num_members):
            self.log_prior.data[i] = self.member_layers[i].log_prior
            self.log_variational_posterior.data[i] = self.member_layers[i].log_variational_posterior

        if self.use_only_elite:
            return xw[self.elite_models]


        return xw


    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, in_size={self.in_size}, "
            f"out_size={self.out_size}, bias={self.use_bias}"
        )

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite


if __name__ == "__main__":
    from mbrl.models.util import EnsembleLinearLayer
    from blitz.losses.kl_divergence import kl_divergence_from_nn
    input_size = 2
    output_size = 2
    ensemble_size = 2
    batch_size = 10
    test_layer = GaussianEnsembleLinearLayer(ensemble_size, input_size, output_size)
    # comp_layer = EnsembleLinearLayer(ensemble_size, input_size, output_size)
    x = torch.Tensor([1,1])
    batch = torch.vstack([(i+1)*x for i in range(batch_size)])
    print(test_layer.get_qw)
    print(test_layer.get_pw)
    out = test_layer.forward(batch)
    print(list(test_layer.parameters()))
    out = test_layer.forward(batch)
    print("-"*20)
    # print(test_layer.log_variational_posterior)
    # print(test_layer.log_prior)
    # print(out)
    print(list(test_layer.parameters()))
