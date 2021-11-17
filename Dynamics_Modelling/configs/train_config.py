
from omegaconf import OmegaConf


"""
Config mostly used for model training, used as input to train function
"""
train_cfg_dict = {
        "model_batch_size": 2048,
        "validation_ratio": .05,
        "shuffle": True,
        "num_epochs_train_model": 5000,
        "patience": 500,
        "lr": 2e-3,
        "weight_decay": 5e-5,
        "target_is_delta": True,
        "autonomus": True, #Used to decide wether to consider actions or not
        "should_log": True,
        "normalize": True
    }

train_cfg = OmegaConf.create(train_cfg_dict)