

from mbrl.models.model_trainer import ModelTrainer
from mbrl.models.one_dim_tr_model import OneDTransitionRewardModel
from mbrl.util.logger import Logger
from mbrl.util.replay_buffer import ReplayBuffer

import mbrl.util.common as common_util


def train(model, train_data, train_cfg, log_dir = None):

    """
    Trains a model on given train_data

    args:
        model -> Model to train
        train_data (dict) -> Dictionary with data of observations and actions,
            each should be of shape (num_data_points, space_dim)
        train_cfg (Omegaconf) -> config for training parameters
        log_dir -> directory to store training logs
    """

    assert model.in_size == (train_data["obs"].shape[1] + train_data["act"].shape[0]), "Model input size does not match train_data"
    assert model.out_size == train_data["obs"].shape[1], "Model output size does not match observation size of train_data"


    if train_cfg.should_log and log_dir is not None:
        logger = Logger(log_dir)
    else:
        logger = None


    wrapper = OneDTransitionRewardModel(model,
         target_is_delta=train_cfg.target_is_delta, normalize=True, learned_rewards=False)
    
    trainer = ModelTrainer(wrapper, optim_lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, logger=logger)
    
    #Wether to use action space or not
    if train_cfg.autonomus:
        train_buffer = ReplayBuffer(train_data["obs"].shape[0]-1, obs_shape=(train_data["obs"].shape[1],), action_shape=(0,))
    else:
        assert train_data["obs"].shape[0] == train_data["act"].shape[0], "Number of observations and actions must be the same"

        train_buffer = ReplayBuffer(train_data["obs"].shape[0]-1, obs_shape=(train_data["obs"].shape[1],), action_shape=(train_data["act"].shape[1],))

    #Add training data to buffer
    for i in range(train_buffer.capacity):
        if train_cfg.autonomus:
            train_buffer.add(train_data["obs"][i,:], 0, train_data["obs"][i+1,:], 0, False)
        else:
            train_buffer.add(train_data["obs"][i,:], train_data["act"][i,:], train_data["obs"][i+1,:], 0, False)

    #Perform training of model
    common_util.train_model_and_save_model_and_data(
        model = wrapper,
        model_trainer = trainer,
        train_cfg = train_cfg.training,
        replay_buffer = train_buffer,
        work_dir = log_dir,
        callback = None
    )

