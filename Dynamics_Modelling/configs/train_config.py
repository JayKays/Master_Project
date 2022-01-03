
from omegaconf import OmegaConf


"""
Config mostly used for model training, used as input to train function
"""

train_cfg_dict = {
        "model_batch_size": 2048,
        "validation_ratio": 0,
        "shuffle": True,
        "num_epochs_train_model": 10000,
        "patience":500,
        "lr": 1e-4,
        "weight_decay": 2.5e-5,
        "target_is_delta": True,
        "autonomus": False, #Used to decide wether to consider actions or not
        "should_log": True,
        "normalize": True,
        "learned_rewards": False
    }

train_cfg = OmegaConf.create(train_cfg_dict)