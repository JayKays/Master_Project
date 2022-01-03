
from time import strftime
from numpy.core.records import array
from util import load_rewrite_buffer, make_model_and_train_data, model_from_cfg, load_buffer, seed_everything
from Data_preperation.filtering import lowpass_fft, lowpass_butter
from Modelling.train import train, train_from_buffer
from configs.train_config import train_cfg
from configs.model_config import model_cfg
from evaluator import DatasetEvaluator
from data_plotting import plot_training_data
import numpy as np
import os
from datetime import datetime


def evaluate(model_dir, data_dir = None, out_dir = None, dyn_model = None, format = "png"):

    if data_dir is None:
        data_dir = model_dir
    
    if out_dir is None:
        out_dir = model_dir + "/evalutaion_test"
    
    evaluator = DatasetEvaluator(model_dir, data_dir, out_dir, model=dyn_model, format = format)
    evaluator.run()


def main():
    seed_everything(42)
    #Set to proper 
    buffer_dir = "32_traj_varying_force"                #Training data path
    model_cfg.type = "PNN"                              #PNN or BNN
    log_dir = f"Model_results/{model_cfg.type}_test"    #Dir to log training data and model
    
    os.makedirs(log_dir)
    print("Log_dir: ", log_dir.split("/")[-1])

    #Define state and action space
    '''
    idx_list:   x   y   z
    Force       0,  1,  2
    Torque      3,  4,  5
    Pos         6,  7,  8
    Vel         9, 10, 11
    Ang.vel     12, 13, 14
    '''
    state_idx = [2, 8, 11, 6, 7]    #VIC states
    state_idx = [2, 8, 11]          #z-states          
    state_idx = None                #None -> use all states
    act_idx = None  

    #Data_processing
    load_rewrite_buffer(buffer_dir, log_dir, state_idx=state_idx, act_idx=act_idx, states_as_act=False)
    train_buffer = load_buffer(log_dir)
    
    model_cfg.input_size = train_buffer.obs.shape[1] + train_buffer.action.shape[1]
    model_cfg.output_size = train_buffer.obs.shape[1]

    print(f"model_input_size = {model_cfg.input_size}")
    print(f"model_output_size = {model_cfg.output_size}")

    #Generate model from cfg
    model = model_from_cfg(model_cfg)
    if model_cfg.type == "BNN":
    
        num_batches = len(train_buffer)//train_cfg.model_batch_size

        if num_batches%len(train_buffer): 
            num_batches += 1
        
        model.num_batches = num_batches
        print("Batch count: ", num_batches)

    #Training
    start = datetime.now()
    wrapper = train_from_buffer(model, train_buffer, train_cfg, model_cfg, log_dir)
    print(f"Training time: {(datetime.now() - start)}")
    evaluate(log_dir, dyn_model=wrapper, format = "svg")


if __name__ == "__main__":
    seed_everything(42)
    main()