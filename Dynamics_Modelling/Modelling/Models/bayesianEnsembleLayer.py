
import torch
from torch import nn
from .weightSampling import TrainableRandomDistribution, PriorWeightDistribution


class BayesianModule(nn.Module):
    '''
    Base Bayesian class to enable methods like .freeze()
    '''
    def __init__(self):
        super().__init__()


#The following class is the BayesianLinearLayer class from blitz, but adapted to be used as a layer in a network ensemble:
#https://github.com/piEsposito/blitz-bayesian-deep-learning/blob/master/blitz/modules/linear_bayesian_layer.py

class BayesianLinearEnsembleLayer(BayesianModule):
    """
    Bayesian Linear Ensemble layer, implements a linear layer as proposed in the Bayes by Backprop paper, which is
    also capable of acting as an efficient linear layer in a bayesian network ensemble.
    
    parameters:
        num_members: int -> size of the network ensemble
        in_fetaures: int -> number  of input features for the layer
        out_features: int -> number of output features for the layer
        bias: bool -> whether the layer should use bias or not, defaults to True
        prior_sigma_1: float -> prior sigma1 on the mixture prior distribution
        prior_sigma_2: float -> prior sigma2 on the mixture prior distribution
        prior_pi: float -> pi on the scaled mixture prior (should be between 0-1)
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not
    """
    def __init__(self,
                 num_members,
                 in_features,
                 out_features,
                 bias=True,
                 prior_sigma_1 = 0.9,
                 prior_sigma_2 = 0.01,
                 prior_pi = 0.7,
                 posterior_mu_init = 0,
                 posterior_rho_init = -7.0,
                 freeze = False,
                 prior_dist = None):
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

        # Variational weight parameters and sampler
        self.weight_mu = nn.Parameter(torch.Tensor(num_members, in_features, out_features).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(num_members, in_features, out_features).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = TrainableRandomDistribution(self.weight_mu, self.weight_rho)

        # Variational bias parameters and sampler
        self.bias_mu = nn.Parameter(torch.Tensor(num_members, 1, out_features).normal_(posterior_mu_init, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(num_members, 1, out_features).normal_(posterior_rho_init, 0.1))
        self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)

        # Prior distributions
        self.weight_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.bias_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        '''
        Computes the forward pass of the Bayesian layer by sampling weights and biases
        and computes the output y = W*x + b

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

        #Calculate forward pass output whith sampled weights and biases
        xw = x.matmul(w) + b

        return xw

    def forward_frozen(self, x):
        """
        Computes the feedforward operation with the expected value for weight and biases
        Equivalent to a deterministic linear forward pass
        """
        if self.bias:
            return x.matmul(self.weight_mu) + self.bias_mu
        else:
            return x.matmul(self.weight_mu)




if __name__ == "__main__":

    enseble_size = 1
    in_size = 2
    out_size = 2

    test_layer = BayesianLinearEnsembleLayer(enseble_size, in_size, out_size) 

    print(list(test_layer.parameters()))