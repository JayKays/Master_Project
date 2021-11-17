
from mbrl.models import one_dim_tr_model
import numpy as np
import torch
from omegaconf.omegaconf import OmegaConf
from Modelling.Models.BNN import BNN
from mbrl.models.gaussian_mlp import GaussianMLP
from mbrl.models.one_dim_tr_model import OneDTransitionRewardModel



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

def get_train_data(file_path, obs_idx, act_idx = None):
    """
    Loads data array from file_path and converts it to train_data dict
    based on ideces in obs_idx and act_idx
    """
    use_act = act_idx is not None

    data_arr = np.load(file_path)
    data_dict = train_data_from_array(data_arr, obs_idx, act_idx, use_act = use_act)
    
    return data_dict

def labeled_data(x, y):
    "Creates dictonary of data with input x and targets y"
    return {'x': x, 'y': y}

def model_from_cfg(model_cfg, model_type = None):
    """
    Initilizes either a BNN or PNN ensemble model base on parameters in model_cfg

    args:
        model_cfg: config with model parameters
        model_type: Type of model to initialize. Needs to be either 'BNN' og 'PNN' or be given in model_cfg
    output: 
        model: torch model with specified parameters
    """

    if model_type is None:
        model_type = model_cfg.get("type", None)

    assert model_type == "BNN" or model_type == "PNN", "Model type must be given in cfg or be either PNN og BNN"

    if model_type == "BNN":
        model_cfg.type = "BNN"
        model = BNN(
            in_size = model_cfg.input_size,
            out_size = model_cfg.output_size,
            device = model_cfg.device,
            num_layers = model_cfg.num_layers,
            propagation_method = model_cfg.propagation_method,
            activation_fn_cfg = model_cfg.activation_fn_cfg,
            ensemble_size = model_cfg.ensemble_size,
            prior_sigma = model_cfg.BNN.prior_sigma,
            prior_pi = model_cfg.BNN.prior_pi,
            freeze = model_cfg.BNN.freeze
        )
        return model

    if model_type == "PNN":
        model_cfg.type = "PNN"
        model = GaussianMLP(
            in_size = model_cfg.input_size,
            out_size = model_cfg.output_size,
            device = model_cfg.device,
            num_layers = model_cfg.num_layers,
            propagation_method = model_cfg.propagation_method,
            activation_fn_cfg = model_cfg.activation_fn_cfg,
            ensemble_size = model_cfg.ensemble_size,
            hid_size = model_cfg.hid_size,
            deterministic = model_cfg.PNN.deterministic
        ) 
        return model 

def load_model(load_dir):

    model_cfg = OmegaConf.load(load_dir + "/model_cfg")
    train_cfg = OmegaConf.load(load_dir + "/train_cfg")

    # if OmegaConf.is_missing(cfg, cfg.type):
    #     model_type = "PNN"
    # else:
    #     model_type = "BNN"
    
    model = model_from_cfg(model_cfg)
    wrapper = OneDTransitionRewardModel(
        model,
        train_cfg.target_is_delta,
        train_cfg.normalize,
    )

    wrapper.load(load_dir)

    return wrapper

def MC_mean_var(model, x, num_samples = 100):
        """
        Computes mean and variance of stoachastic model prediction
        through monte carlo sampling
        """

        # shape = (num_samples, Ensemble_size, Batch_size, output_size)
        samples = torch.zeros((num_samples, len(model), x.shape[0], model.out_size))
        
        for n in range(num_samples):
            samples[n,...] = model(x)
        
        mean = torch.mean(samples[:,:,:,:], dim = 0)
        var = torch.var(samples[:,:,:,:], dim = 0)

        return mean, var, #torch.log(var)

def create_test_data(train_size = 2000):

    """
    Generates som random noisy data sampled from a sine wave
    """

    num_points = 10000
    x_data = np.linspace(-12, 12, num_points)
    y_data = np.sin(x_data)

    x_train = np.zeros((train_size*2, 1))
    y_train = np.zeros((train_size*2, 1))

     #Half with lower noise
    train_val_idx_1 = np.random.choice(list(range(1200, 3500)), 
                                    size=train_size, 
                                    replace=False)
    mag = 0.05
    x_train[:train_size,0] = x_data[train_val_idx_1[:train_size]]
    y_train[:train_size,0] = y_data[train_val_idx_1[:train_size]] + mag * np.random.randn(train_size)

    # Half with higher noise
    train_val_idx_2 = np.random.choice(list(range(6500, 8800)), 
                                    size=train_size, 
                                    replace=False)
    mag = 0.20
    x_train[train_size:,0] = x_data[train_val_idx_2[:train_size]]
    y_train[train_size:,0] = y_data[train_val_idx_2[:train_size]] + mag * np.random.randn(train_size)

    indeces = np.random.choice(train_size*2, train_size*2, replace=False)

    return x_data, y_data, x_train[indeces,:], y_train[indeces,:]

