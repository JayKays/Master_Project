

from mbrl.util.replay_buffer import ReplayBuffer
import numpy as np
import torch
from omegaconf.omegaconf import OmegaConf
# from torch._C import int32
from Modelling.Models.BNN import BNN
from mbrl.models.gaussian_mlp import GaussianMLP
from mbrl.models.one_dim_tr_model import OneDTransitionRewardModel
from Data_preperation.filtering import filter_med_lowpass

#TODO: Change model cfg setup to use mbrl cfg structure for utility function support


def make_model_and_train_data(model_cfg, data_arr, obs_idx, act_idx = None, use_act = False):

    train_data = train_data_from_array(data_arr, obs_idx, act_idx, use_act=use_act)

    if use_act:
        model_cfg.input_size = train_data["obs"].shape[1] + train_data["act"].shape[1]
    else:
        model_cfg.input_size = train_data["obs"].shape[1]
    model_cfg.output_size = train_data["obs"].shape[1]

    model = model_from_cfg(model_cfg)

    return model, train_data, model_cfg

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
        learned_rewards=train_cfg.get("learned_rewards", False)
    )

    wrapper.load(load_dir)

    return wrapper

def monte_carlo_mean_var(model, x, num_samples = 100):
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

def load_buffer(load_dir):

    data = np.load(load_dir + "/replay_buffer.npz")

    traj_idx = data["trajectory_indices"]
    if len(traj_idx) > 0:
        traj_length = traj_idx[0,1] - traj_idx[0,0]
    else:
        traj_length = None

    dataset_size = data["obs"].shape[0]
    obs_shape = (data["obs"].shape[1],)
    act_shape = (data["action"].shape[1],)


    train_buffer = ReplayBuffer(
        capacity=dataset_size,
        obs_shape=obs_shape,
        action_shape=act_shape,
        max_trajectory_length= traj_length
    )

    train_buffer.load(load_dir)

    return train_buffer

def load_rewrite_buffer(load_dir, save_dir, state_idx = None, act_idx = None, states_as_act = False, traj_clip = 100, filter = True, cut_first_last_traj = True):
    data = np.load(load_dir + "/replay_buffer.npz")

    traj_idx = data["trajectory_indices"]
    if len(traj_idx) > 0:
        traj_length = traj_idx[0,1] - traj_idx[0,0]
    else:
        traj_length = None
        
    data_idx = cut_off_traj_starts_idx(data["obs"].shape[0], cutoff_length=traj_clip, traj_length = traj_length)


    if state_idx is None:
        state_idx = np.arange(data["obs"].shape[1])
    
    if states_as_act and act_idx is not None:
        act = data["obs"][data_idx]
    elif act_idx is None:
        act_idx = np.arange(data["action"].shape[1])
        act = data["action"][data_idx]
    else:
        act = data["action"][data_idx]
    
      
    obs = data["obs"][data_idx]
    next_obs = data["next_obs"][data_idx]
    reward = data["reward"][data_idx]
    done = data["done"][data_idx]

    if filter:
        obs = filter_med_lowpass(obs)
        act = filter_med_lowpass(act)
        next_obs = filter_med_lowpass(next_obs)

    if cut_first_last_traj and traj_length is not None:
        new_traj_length = traj_length - traj_clip
        
        obs = obs[new_traj_length : -new_traj_length]
        act = act[new_traj_length : -new_traj_length]
        next_obs = next_obs[new_traj_length : -new_traj_length]
        reward = reward[new_traj_length : -new_traj_length]
        done = done[new_traj_length : -new_traj_length]



    traj_indices = []
    if traj_length is not None:
        cur_idx = 0

        for _ in range(len(traj_idx)):
            traj_indices.append([int(cur_idx), int(cur_idx + traj_length - traj_clip)])
            cur_idx += traj_length - traj_clip

    # print(traj_indices or [])
    np.savez(
            save_dir + "/replay_buffer.npz",
            obs=obs[:,state_idx],
            next_obs=next_obs[:,state_idx],
            action=act[:,act_idx],
            reward=reward,
            done=done,
            trajectory_indices=traj_indices,
        )


def cut_off_traj_starts_idx(total_length, cutoff_length = 100, traj_length = 1500):

    if traj_length is None:
        return np.arange(total_length)

    num_traj = total_length//traj_length

    idx = np.arange(total_length)

    for traj in range(num_traj):
        idx[traj*traj_length:traj*traj_length+ cutoff_length] = -1

    return idx[idx != -1]

def seed_everything(seed: int):
    '''
    https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    '''
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    pass

