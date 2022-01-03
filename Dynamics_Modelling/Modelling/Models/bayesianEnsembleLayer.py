
import torch
import numpy as np
from torch import nn
from .weightSampling import TrainableRandomDistribution, PriorWeightDistribution


class BayesianLinearEnsembleLayer(nn.Module):
    """
    Bayesian Linear Ensemble layer, implements a linear layer as proposed in the Bayes by Backprop paper. 
    This is also capable of acting as an efficient linear layer in a bayesian network ensemble, given num_members > 1.
    
    parameters:
        num_members(int): size of the network ensemble
        in_fetaures(int): number  of input features for the layer
        out_features(int): number of output features for the layer
        bias(bool): whether the layer should use bias or not, defaults to True
        prior_sigma_1 (float): sigma1 of the  gaussian mixture prior distribution
        prior_sigma_2 (float): sigma2 of the gaussian mixture prior distribution
        prior_pi(float): scaling factor of the guassian mixture prior (must be between 0-1)
        posterior_mu_init (float): mean of the mu parameter init
        posterior_rho_init (float): mean of the rho parameter init
        freeze(bool): wheter the model will start with frozen(deterministic) weights, or not
        prior_dist: A potential prior distribution of weghts
        truncated_init(bool): Wether to use a truncated normal distribution for parameter init or not
        moped (bool): Wether or not to use the moped initialization of weight std (rho = delta*|w|)
        delta(float): The delta value in moped init, not used if moped = False 
    """
    def __init__(self,
                 num_members,
                 in_features,
                 out_features,
                 bias=True,
                 prior_sigma_1 = 0.9,
                 prior_sigma_2 = 0.001,
                 prior_pi = 0.7,
                 posterior_mu_init = 0,
                 posterior_rho_init = -7.0,
                 freeze = False,
                 prior_dist = None,
                 truncated_init = True,
                 delta = 0.1,
                 moped = False):
        super().__init__()

        #main layer parameters
        self.num_members = num_members
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.freeze = freeze

        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        #parameters for the scale mixture gaussian prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        #Parameters of truncated normal distribution for parameter initialization
        init_std = 1 / (2*np.sqrt(self.in_features))
        init_std = 0.1
        mu_init_max, mu_init_min = posterior_mu_init + 2*init_std, posterior_mu_init-2*init_std
        rho_init_max, rho_init_min = posterior_rho_init + 2*init_std, posterior_rho_init-2*init_std

        # Variational parameters and sampler for weights and biases
        if truncated_init:
            self.weight_mu = nn.Parameter(torch.Tensor(num_members, in_features, out_features).normal_(posterior_mu_init, init_std).clamp(mu_init_min, mu_init_max))
            self.bias_mu = nn.Parameter(torch.Tensor(num_members, 1, out_features).normal_(posterior_mu_init, init_std).clamp(mu_init_min, mu_init_max))
        else:
            self.weight_mu = nn.Parameter(torch.Tensor(num_members, in_features, out_features).normal_(posterior_mu_init, init_std))
            self.bias_mu = nn.Parameter(torch.Tensor(num_members, 1, out_features).normal_(posterior_mu_init, init_std))

        if moped:
            self.weight_rho = nn.Parameter(torch.log(torch.expm1(delta * torch.abs(self.weight_mu.data))))
            self.bias_rho = nn.Parameter(torch.log(torch.expm1(delta * torch.abs(self.bias_mu.data))))
        else:
            self.weight_rho = nn.Parameter(torch.Tensor(num_members, in_features, out_features).normal_(posterior_rho_init, init_std))
            self.bias_rho = nn.Parameter(torch.Tensor(num_members, 1, out_features).normal_(posterior_rho_init, init_std))

        self.weight_sampler = TrainableRandomDistribution(self.weight_mu, self.weight_rho)
        self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)

        # Prior distributions
        self.weight_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.bias_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        '''
        Computes the forward pass of the Bayesian layer by sampling weights and biases
        and computes the output y = W*x + b with the sampled paramters.

        returns:
            torch.tensor with shape [num_members, out_features]
        '''
        
        #if the model is frozen, return deterministic forward pass
        if self.freeze:
            return self.forward_frozen(x)

        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = torch.zeros((self.out_features))
            b_log_posterior = 0
            b_log_prior = 0

        # Get the complexity cost
        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        #Calculate forward pass output whith sampled weights (and biases)
        xw = x.matmul(w) + b

        return xw

    def forward_frozen(self, x):
        '''
        Computes the deterministic forward pass using only the means of the weight distributions
        
        returns:
            torch.tensor with shape [num_members, out_features]
        '''

        if self.bias:
            return x.matmul(self.weight_sampler.mu) + self.bias_sampler.mu
        else:
            return x.matmul(self.weight_sampler.mu)




if __name__ == "__main__":

    enseble_size = 1
    in_size = 2
    out_size = 2

    test_layer = BayesianLinearEnsembleLayer(enseble_size, in_size, out_size) 

    print(list(test_layer.parameters()))