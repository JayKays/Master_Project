# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, List

import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.pyplot import plot
from mbrl.types import ModelInput
import numpy as np
import pathlib
import torch

import mbrl.util
import mbrl.util.common
import mbrl.util.mujoco
from torch._C import has_spectral

from util import load_model
from mbrl.models.gaussian_mlp import GaussianMLP


class DatasetEvaluator:
    def __init__(self, model_dir: str, dataset_dir: str, output_dir: str, model = None, format = "png"):
        self.model_path = pathlib.Path(model_dir)
        self.output_path = pathlib.Path(output_dir + f"_{format}")
        pathlib.Path.mkdir(self.output_path, parents=True, exist_ok=True)

        if model is None:
            self.dynamics_model = load_model(model_dir)
        else:
            self.dynamics_model = model

        if hasattr(self.dynamics_model.model, "freeze_model"):
            self.dynamics_model.model.freeze_model()

        data = np.load(dataset_dir + "/replay_buffer.npz")

        self.replay_buffer = mbrl.util.ReplayBuffer(
            capacity= data["obs"].shape[0],
            obs_shape= (data["obs"].shape[1],),
            action_shape= (data["action"].shape[1],) 
        )
        self.replay_buffer.load(dataset_dir)
        self.format = format
        self.label_size = 15

    def plot_dataset_results(self, dataset: mbrl.util.TransitionIterator):
        print("Generating dataset prediction plots for whole dataset")
        plot_path = pathlib.Path(self.output_path / "evaluations")
        pathlib.Path.mkdir(plot_path, parents=True, exist_ok=True)

        all_means: List[np.ndarray] = []
        all_targets = []

        # Iterating over dataset and computing predictions
        for batch in dataset:
            (
                outputs,
                target,
            ) = self.dynamics_model.get_output_and_targets(batch)
            
            if isinstance(outputs, tuple):
                all_means.append(outputs[0].cpu().numpy())
            else:
                all_means.append(outputs.cpu().numpy())

            all_targets.append(target.cpu().numpy())
            
        # Consolidating targets and predictions
        all_means_np = np.concatenate(all_means, axis=-2)
        targets_np = np.concatenate(all_targets, axis=0)

        if all_means_np.ndim == 2:
            all_means_np = all_means_np[np.newaxis, :]
        assert all_means_np.ndim == 3  # ensemble, batch, target_dim

        # Visualization
        num_dim = targets_np.shape[1]
        for dim in range(num_dim):

            plt.figure(figsize=(8, 8))

            self.eval_plot(all_means_np[...,dim], targets_np[:,dim], sample_factor=20)

            fname = plot_path / f"pred_dim{dim}.{self.format}"
            plt.savefig(fname, format = self.format)#, format = "svg")
            plt.close()

    def run(self):
        batch_size = 32
        if hasattr(self.dynamics_model, "set_propagation_method"):
            self.dynamics_model.set_propagation_method(None)
            # Some models (e.g., GaussianMLP) require the batch size to be
            # a multiple of number of models
            batch_size = len(self.dynamics_model) * 8
        dataset, _ = mbrl.util.common.get_basic_buffer_iterators(
            self.replay_buffer, batch_size=batch_size, val_ratio=0
        )
        
        # self.plot_rollout(horizon = 5)
        self.plot_one_step_predictions(dataset)
        self.plot_dataset_results(dataset)
        self.plot_multiple(dataset)

    def plot_rollout(self, horizon  = None):
        print(f"Generating trajectory rollout plots, with horizon {horizon}")
        plot_path = pathlib.Path(self.output_path / "trajectories")
        pathlib.Path.mkdir(plot_path, parents=True, exist_ok=True)

        traj = self.replay_buffer.sample_trajectory()
        obs = traj.obs
        next_obs = traj.next_obs
        act = traj.act

        horizon = horizon if horizon is not None else len(traj)
        
        prop_traj = np.zeros((horizon+1, obs.shape[1]))
        prop_traj[0,:] = obs[0,:]
        with torch.no_grad():
            for i in range(1, horizon+1):
                observation = torch.tensor(prop_traj[i-1,:])
                action = torch.tensor(act[i-1,:])
                model_input,_,_ = self.dynamics_model._get_model_input(observation, action)
                output = self.dynamics_model(model_input.float())

                if isinstance(output, tuple):
                    pred = output[0]
                else:
                    pred = output

                prop_traj[i,:] = prop_traj[i-1,:] + pred.mean(dim=0)[0].cpu().numpy()
            
        num_dim = obs.shape[1]
        for dim in range(num_dim):

            plt.figure()

            plt.plot(
                obs[:horizon,dim],
                color='k',
                label="Observed Trajectory"
            )

            plt.plot(
                prop_traj[:horizon,dim],
                color = 'r',
                label = "Predicted Trajectory"
            )
            plt.legend(prop = {'size': 11})
            # plt.title("Trajectory Propagation")
            fname = plot_path / f"pred_dim{dim}.{self.format}"
            plt.savefig(fname)
            plt.close()


    def plot_one_step_predictions(self, dataset):
        print("Generating one step prediciton plots over whole dataset")
        plot_path = pathlib.Path(self.output_path / "one_step_predictions")
        pathlib.Path.mkdir(plot_path, parents=True, exist_ok=True)

        all_means: List[np.ndarray] = []
        all_targets = []
        all_std: List[np.ndarray] = []

        # Iterating over dataset and computing predictions
        for batch in dataset:
        
            model_input, target = self.dynamics_model._process_batch(batch)

            outputs, var_epi, var_ale = pred_uncertainty(self.dynamics_model, model_input)

            stds = np.sqrt(var_epi + var_ale)

            if isinstance(outputs, tuple):
                all_means.append(outputs.cpu().numpy())
            else:
                all_means.append(outputs.cpu().numpy())

            all_targets.append(target.cpu().numpy())
            all_std.append(stds)
            
        # Consolidating targets and predictions
        all_means_np = np.concatenate(all_means, axis=-2)
        targets_np = np.concatenate(all_targets, axis=0)
        std_np = np.concatenate(all_std, axis = 0)

        if all_means_np.ndim == 2:
            all_means_np = all_means_np[np.newaxis, :]
        assert all_means_np.ndim == 3  # ensemble, batch, target_dim
        
        num_dim = targets_np.shape[1]
        for dim in range(num_dim):
            sort_idx = targets_np[:, dim].argsort()
            means = all_means_np[..., sort_idx, dim]
            target = targets_np[sort_idx, dim]
            std = std_np[sort_idx, dim]

            plt.figure(figsize=(16,8))
            self.one_step_plot(means, target, std)
            fname = plot_path/ f"pred_dim{dim}.{self.format}"
            plt.savefig(fname, format = self.format)#, format = "svg")
            plt.close()


    def plot_multiple(self, dataset):
        print("Generating combination plots")
        plot_path = pathlib.Path(self.output_path / "Evaulation_combs")
        pathlib.Path.mkdir(plot_path, parents=True, exist_ok=True)

        all_means: List[np.ndarray] = []
        all_targets = []
        all_std_epi: List[np.ndarray] = []
        all_std_ale: List[np.ndarray] = []
        # Iterating over dataset and computing predictions
        for batch in dataset:
            # (
            #     outputs,
            #     target,
            # ) = self.dynamics_model.get_output_and_targets(batch)
            model_input, target = self.dynamics_model._process_batch(batch)

            means, var_epi, var_ale = pred_uncertainty(self.dynamics_model, model_input)

            all_means.append(means.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_std_epi.append(np.sqrt(var_epi))
            all_std_ale.append(np.sqrt(var_ale))
            
        # Consolidating targets and predictions
        all_means_np = np.concatenate(all_means, axis=-2)
        targets_np = np.concatenate(all_targets, axis=0)
        std_epi_np = np.concatenate(all_std_epi, axis = 0)
        std_ale_np = np.concatenate(all_std_ale, axis = 0)

        if all_means_np.ndim == 2:
            all_means_np = all_means_np[np.newaxis, :]
        assert all_means_np.ndim == 3  # ensemble, batch, target_dim

        num_dim = targets_np.shape[1]
        for dim in range(num_dim):
            sort_idx = targets_np[:, dim].argsort()
            means = all_means_np[..., sort_idx, dim]
            target = targets_np[sort_idx, dim]
            std_ale = std_ale_np[sort_idx, dim]
            std_epi = std_epi_np[sort_idx,dim]
            

            mean_of_means = means.mean(0)
            ymin1, ymax1 = np.min(mean_of_means - 2*std_ale), np.max(mean_of_means + 2*std_ale)
            ymin2, ymax2 = np.min(mean_of_means - 2*std_epi), np.max(mean_of_means + 2*std_epi)

            ymin = min(ymin1, ymin2)*1.2
            ymax = max(ymax1, ymax2)*1.2

            plt.figure(figsize=(16,8))

            plt.subplot2grid((2,4), (0,0), colspan=2)
            self.one_step_plot(means, target, std_epi)
            plt.ylim([ymin, ymax])
            plt.title("One step predictions (Epistemic)", fontsize = self.label_size+2)

            plt.subplot2grid((2,4), (1,0), colspan=2)
            self.one_step_plot(means, target, std_ale, no_legend=True)
            plt.ylim([ymin, ymax])
            plt.title("One step predictions (Aleatoric)", fontsize = self.label_size+2)
            
            plt.subplot2grid((2,4), (0,2), rowspan=2, colspan=2)
            self.eval_plot(means, target)
            plt.title("Ensemble predictions", fontsize = self.label_size+2)

            plt.tight_layout(pad = 5)

            fname = plot_path / f"pred_dim{dim}.{self.format}"
            plt.savefig(fname, format = self.format)
            plt.close()

    def one_step_plot(self, means, targets, std, no_legend = False, subsample = False):
        mean_of_means = means.mean(0)

        
        plt.fill_between(np.arange(mean_of_means.shape[0]),mean_of_means - 2*std, mean_of_means + 2*std, color='#b8b8f8', label = "Uncertainty")
       
        plt.plot(
            mean_of_means,
            color="r",
            linewidth=0.5,
            label = "Ensemble prediction"
        )

        plt.plot(
            targets,
            color="k",
            linewidth=0.5,
            label = "Ground Truth"
        )
        plt.ylabel("Prediction", fontsize = self.label_size)
        if not no_legend:
            plt.legend(loc = "upper left", prop = {'size': 11})
    
    def eval_plot(self, all_means, targets, sample_factor = 40):
        sort_idx = targets.argsort()
        subsample_size = len(sort_idx) // sample_factor + 1
        subsample = np.random.choice(len(sort_idx), size=(subsample_size,))
        means = all_means[..., sort_idx][..., subsample]  # type: ignore
        target = targets[sort_idx][subsample]
        
        for i in range(all_means.shape[0]):
            if not i:
                plt.plot(target, means[i], ".", markersize=2, label = "Single model predictions")
            else:
                plt.plot(target, means[i], ".", markersize=2)
        mean_of_means = means.mean(0)
        mean_sort_idx = target.argsort()

        plt.plot(
            target[mean_sort_idx],
            mean_of_means[mean_sort_idx],
            color="r",
            linewidth=0.5,
            label = "Ensemble prediction"
        )
        
        plt.plot(
            target,
            target,
            # linewidth=2,
            color="k",
            label = "Target"
        )
        plt.legend(loc = "upper left", prop = {'size': 11})
        plt.xlabel("Target", fontsize = self.label_size)
        plt.ylabel("Prediction", fontsize = self.label_size)
        
def monte_carlo_mean_var(model, x, num_samples = 10):
        """
        Computes mean and variance of stoachastic model prediction
        through monte carlo sampling
        """

        # shape = (num_samples, Ensemble_size, Batch_size, output_size)
        if hasattr(model.model, "unfreeze_model"):
            model.model.unfreeze_model()
            # print("Unfrozen BNN for eval")
        
        with torch.no_grad():
            samples = torch.zeros((num_samples, len(model), x.shape[0], model.model.out_size))
            
            for n in range(num_samples):
                    samples[n,...] = model(x)
            
            mean = torch.mean(samples[:,:,:,:], dim = 0)
            var = torch.var(samples[:,:,:,:], dim = 0)

        if hasattr(model, "freeze_model"):
            model.model.freeze_model()
        
        return mean, var
    
def pred_uncertainty(model, x):
    '''
    Predicts the uncertainty of a given models predictions on input x,

    returns:
        mean: Prediciton means
        std: Prediction standard deviations
    '''

    if isinstance(model.model, GaussianMLP):
        with torch.no_grad():
            mean, pred_logvar = model(x)
            pred_var = pred_logvar.exp()

    else:

        mean, pred_var =  monte_carlo_mean_var(model, x)

    var_epi = mean.var(dim = 0).cpu().numpy()
    var_ale = pred_var.mean(dim = 0).cpu().numpy()

    std = np.sqrt(var_epi + var_ale)

    return mean, var_epi, var_ale


def eliminate_outliers(data, m = 1):
    """
    returns array of indeces in data which lies in the confidence interval given by m
    """

    idx = np.where(np.abs(data - np.median(data) < m * np.std(data)))

    return idx

