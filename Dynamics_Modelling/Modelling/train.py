

from mbrl.models.model_trainer import ModelTrainer
from mbrl.models.one_dim_tr_model import OneDTransitionRewardModel
from mbrl.util.logger import Logger
from mbrl.util.replay_buffer import ReplayBuffer

import mbrl.util.common as common_util


def train(model, train_data, cfg):

    if cfg.log.logging:
        logger = Logger(cfg.log.log_dir)
    else:
        logger = None


    wrapper = OneDTransitionRewardModel(model,
         target_is_delta=cfg.model.target_is_delta, normalize=True, learned_rewards=False)
    
    trainer = ModelTrainer(wrapper, optim_lr=cfg.training.lr, weight_decay=cfg.training.weight_decay, logger=logger)

    train_buffer = ReplayBuffer(train_data.shape[0]-1, obs_shape=(train_data.shape[1],), action_shape=(0,))

    for i in range(train_buffer.capacity):
        train_buffer.add(train_data[i,:], 0, train_data[i+1,:], 0, False)

    common_util.train_model_and_save_model_and_data(
        model = wrapper,
        model_trainer = trainer,
        cfg = cfg.training,
        replay_buffer = train_buffer,
        work_dir = cfg.log.log_dir,
        callback = None
    )

