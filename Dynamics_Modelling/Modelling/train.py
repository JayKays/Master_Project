

from mbrl.models.model_trainer import ModelTrainer
from mbrl.models.one_dim_tr_model import OneDTransitionRewardModel
from mbrl.util.logger import Logger
from mbrl.util.replay_buffer import ReplayBuffer
from mbrl.models.gaussian_mlp import GaussianMLP
from numpy import log
from omegaconf import OmegaConf

import mbrl.util.common as common_util
import os
import numpy as np
from .Models.BNN import BNN

def train(model, train_data, train_cfg, model_cfg, log_dir = None):

    """
    Trains a model on given train_data

    args:
        model: Model to train
        train_data (dict): Dictionary with data of observations and actions,
            each should be of shape (num_data_points, space_dim)
        train_cfg (Omegaconf): config for training parameters
        log_dir: directory to store training logs
    """
    labeled = ("x" in train_data and "y" in train_data)


    if labeled:
        assert model.in_size == train_data["x"].shape[1], "Input data size does not match model input size"
        assert model.out_size == train_data["y"].shape[1], "Labels size does not match model output size"
        train_cfg.target_is_delta = False
    else:
        assert "obs" in train_data and "act" in train_data, "Using unlabeled data requires both observations and actions"
        assert model.out_size == train_data["obs"].shape[1], "Model output size does not match observation size of train_data"
        if not train_cfg.autonomus:
            assert model.in_size == (train_data["obs"].shape[1] + train_data["act"].shape[1]), "Model input size does not match train_data"
        else:
            assert model.in_size == train_data["obs"].shape[1], "Model input size does not match train_data"

    if log_dir is not None and not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        print("Created log dir: ", log_dir)


    if train_cfg.should_log and log_dir is not None:
        logger = Logger(log_dir)
    else:
        logger = None

    wrapper = OneDTransitionRewardModel(model,
         target_is_delta = train_cfg.target_is_delta, normalize=train_cfg.normalize, learned_rewards=False)
    
    trainer = ModelTrainer(wrapper, optim_lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, logger=logger)
    
    if labeled:
        train_buffer = ReplayBuffer(train_data["x"].shape[0], obs_shape=(train_data["x"].shape[1],), action_shape=(0,))

        for i in range(train_buffer.capacity):
            train_buffer.add(train_data["x"][i,:], 0, train_data["y"][i,:], 0, False)
    else:
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

    #Save model config for loading
    if OmegaConf.is_missing(model_cfg, "type"):
        if isinstance(model, BNN):
            model_cfg.type = "BNN"
        else:
            model_cfg.type = "PNN"
        
    OmegaConf.save(model_cfg, log_dir + "/model_cfg")
    OmegaConf.save(train_cfg, log_dir + "/train_cfg")
    train_buffer.save(log_dir)

    #Train and save trained model
    common_util.train_model_and_save_model_and_data(
        model = wrapper,
        model_trainer = trainer,
        cfg = train_cfg,
        replay_buffer = train_buffer,
        work_dir = log_dir,
        callback = None
    )

def train_from_buffer(model, train_buffer, train_cfg, model_cfg, log_dir = None):

    """
    Trains a using the given data in train_buffer

    args:
        model: Pytorch model to train
        train_data (replayBuffer): replayBuffer of training data (will be split in train/val) 
        train_cfg (Omegaconf): config for training parameters
        model_cfg (OmegaConf): config for the model (saved to help with loading)
        log_dir: directory to log training and save the trained model
    """

    if log_dir is not None and not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        print("Created log dir: ", log_dir)


    if train_cfg.should_log and log_dir is not None:
        logger = Logger(log_dir)
    else:
        logger = None
    
    wrapper = OneDTransitionRewardModel(model,
         target_is_delta = train_cfg.target_is_delta, normalize=train_cfg.normalize, learned_rewards=False)
    
    trainer = ModelTrainer(wrapper, optim_lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, logger=logger)

    #Save model config for loading
    if OmegaConf.is_missing(model_cfg, "type"):
        if isinstance(model, BNN):
            model_cfg.type = "BNN"
        else:
            model_cfg.type = "PNN"
        
    OmegaConf.save(model_cfg, log_dir + "/model_cfg")
    OmegaConf.save(train_cfg, log_dir + "/train_cfg")
    # train_buffer.save(log_dir)

    #Train and save trained model
    # common_util.train_model_and_save_model_and_data(
    #     model = wrapper,
    #     model_trainer = trainer,
    #     cfg = train_cfg,
    #     replay_buffer = train_buffer,
    #     work_dir = log_dir,
    #     callback = None
    # )

    dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
        train_buffer,
        train_cfg.model_batch_size,
        train_cfg.validation_ratio,
        ensemble_size=len(model),
        shuffle_each_epoch=True,
    )
    if hasattr(model, "update_normalizer"):
        model.update_normalizer(train_buffer.get_all())

    train_loss, val_score = trainer.train(
                            dataset_train,
                            dataset_val=dataset_val,
                            num_epochs=train_cfg.get("num_epochs_train_model", None),
                            patience=train_cfg.get("patience", 1),
                            improvement_threshold=train_cfg.get("improvement_threshold", 0.01),
                            callback=None,
                        )
    np.savez(log_dir + "/train_data.npz", train_loss = train_loss, val_score = val_score)
    wrapper.save(log_dir)
    return wrapper

