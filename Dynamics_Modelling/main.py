
from numpy.core.records import array
from util import make_model_and_train_data
from Data_preperation.filtering import lowpass_fft, lowpass_butter
from Modelling.train import train
from configs.train_config import train_cfg
from configs.model_config import model_cfg
from evaluator import DatasetEvaluator
from data_plotting import plot_training_data
import numpy as np

'''
idx_list: 
t       0
Force   1,  2,  3
Torque  4,  5,  6
Pos     7,  8,  9
Vel     10, 11, 12
Ang.vel 13, 14, 15
'''

def evaluate(model_dir, data_dir = None, out_dir = None):

    if data_dir is None:
        data_dir = model_dir
    
    if out_dir is None:
        out_dir = model_dir + "/evalutaion"
    
    evaluator = DatasetEvaluator(model_dir, data_dir, out_dir)
    evaluator.run()


def main():
    data_file = "single_traj.npy"
    model_cfg.type = "BNN"
    log_dir = f"Logs/{model_cfg.type}_single_traj_64h_filter_4"

    data_dir = "Data_preperation/training_data/"
    data_path = data_dir + data_file
    
    data_arr = np.load(data_path)
    lowpass_butter(data_arr, fs=100, cutoff=4, idx = [3, 12])

    model, train_data, m_cfg = make_model_and_train_data(model_cfg, data_arr[50:,:], [3,9,12])

    train(model, train_data, train_cfg, m_cfg, log_dir = log_dir)
    evaluate(log_dir)
    plot_training_data(log_dir)


if __name__ == "__main__":
    main()
    # evaluate("Logs/BNN_single_traj_filter")
    # evaluate("Logs/PNN_sine_test")
    # plot_training_data("Logs/BNN_single_traj_5l_512h_filter_4")