
from omegaconf import OmegaConf
# import omegaconf

cfg_dict = {
    "data": {
        "obs_shape": (1,),
        "act_shape": (0,)
    },

    "training": {
        "model_batch_size": 32,
        "validation_ratio": .05,
        "shuffle": True,
        "num_epochs_train_model": 50,
        "patience": 20,
        "lr": 1e-2,
        "weight_decay": 1e-4
    },

    "log": {
        "logging": True,
        "log_dir": "Dynamics_Modelling/Logs"
    },

    "model":{
        "target_is_delta": False
    }
}
'''
-model_batch_size (int)
-validation_ratio (float)
-num_epochs_train_model (int, optional)
-patience (int, optional)
-bootstrap_permutes (bool, optional)
'''


cfg = OmegaConf.create(cfg_dict)
