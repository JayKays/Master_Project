
import numpy as np
from Modelling.Models.BNN import BNN
from mbrl.models.gaussian_mlp import GaussianMLP

def train_data_from_array(data_arr, obs_idx = None, act_idx = None, use_act = False):

    """
    Creates a dictionary of observations and actions used as train_data in train fucntion

    args:
        data_arr: npy array of data, shape (num_samples, )
        obs_idx: list of indeces in data_arr to use as observations
        act_idx: list of indeces in data_arr to use as actions
        use_act: wether to use actions or not

    output:
        train_data (dict): dict of observations and actions
    """

    train_data = {
        "obs": np.array([]),
        "act": np.array([])
    }

    if use_act and act_idx is not None:
        train_data["act"] = data_arr[:,act_idx]
    
    if obs_idx is not None:
        train_data["obs"] = data_arr[:,obs_idx]

    return train_data


def model_from_cfg(model_cfg, model_type = None):
    """
    Initilizes either a BNN or PNN ensemble model base on parameters in model_cfg

    args:
        model_cfg: config with model parameters
        model_type: Type of model to initialize, needs to be either 'BNN' og 'PNN'
    output: 
        model: torch model with specified parameters
    """
    assert model_type == "BNN" or model_type == "PNN", "Model type must either be PNN og BNN"

    if model_type == "BNN":
        model = BNN(
            in_size = model_cfg.input_size,
            out_size = model_cfg.output_size,
            device = model_cfg.device,
            num_layers = model_cfg.num_layers,
            propagation_method = model_cfg.propagation_method,
            ensemble_size = model_cfg.BNN.ensemble_size,
            activation_fn_cfg = model_cfg.activation_fn_cfg,
            prior_sigma = model_cfg.BNN.prior_sigma,
            prior_pi = model_cfg.BNN.prior_pi,
            freeze = model_cfg.BNN.freeze
        )
        return model

    if model_type == "PNN":
        model = GaussianMLP(
            in_size = model_cfg.input_size,
            out_size = model_cfg.output_size,
            device = model_cfg.device,
            num_layers = model_cfg.num_layers,
            propagation_method = model_cfg.propagation_method,
            ensemble_size = model_cfg.PNN.ensemble_size,
            activation_fn_cfg = model_cfg.activation_fn_cfg,
            hid_size = model_cfg.PNN.hid_size,
            deterministic = model_cfg.PNN.deterministic
        ) 
        return model 

