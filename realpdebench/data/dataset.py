import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import os

class RealDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        dataset_root,
        dataset_type,
        mode,
        test_mode,
        mask_prob,
        in_step,
        out_step,
        N_autoregressive,
        interval,
        train_ratio,
        split_numerical,
        trunk_length,
        noise_scale,
        n_sim_in_distribution,
        n_sim_out_distribution, 
        n_sim_frame,
        sub_s_real=1,
        sub_s_numerical=1,
        noise_type='gaussian',
        optical_kernel_size=4,
        optical_sigma=1.0
    ):
        super().__init__()
        '''
        dataset_name: name of the dataset
        dataset_root: root path of the dataset
        dataset_type: real | numerical
        mode: train | val | test
        test_mode: all | in_dist | out_dist | seen | unseen
        mask_prob: probability of masking the unmeasured modalities, only for numerical 
                datasets
        in_step: number of steps for input
        out_step: number of steps for output
        N_autoregressive: number of autoregressive times
        interval: interval of the sliding window
        train_ratio: ratio of training data
        split_numerical: split numerical data into training, validation and test data or not
        trunk_length: length of the trunk for splitting on simulation into training and 
                test data
        noise_scale: scale of the noise added to numerical data
        n_sim_in_distribution: number of simulations for in-distribution test
        n_sim_out_distribution: number of simulations for out-distribution test
        n_sim_frame: number of frames in each simulation
        sub_s_real: spatial sub-sampling factor for real data
        sub_s_numerical: spatial sub-sampling factor for numerical data
        noise_type: type of noise, gaussian | poisson | optical
        optical_kernel_size: size of the kernel for optical noise
        optical_sigma: standard deviation of the kernel for optical noise
        '''
        self.dataset_dir = os.path.join(dataset_root, dataset_name)
        self.dataset_path = os.path.join(self.dataset_dir, dataset_type)
        self.dataset_type = dataset_type
        self.mask_prob = mask_prob
        self.noise_scale = noise_scale
        self.noise_type=noise_type
        self.optical_kernel_size = optical_kernel_size
        self.optical_sigma = optical_sigma
        
        self.mode = mode
        self.test_mode = test_mode

        self.in_step = in_step
        self.out_step = out_step * N_autoregressive
        self.N_autoregressive = N_autoregressive
        self.interval = interval
        self.horizon = self.in_step + self.out_step
        self.n_sim_frame = n_sim_frame
        self.trunk_length = trunk_length
        self.sub_s_real = sub_s_real
        self.sub_s_numerical = sub_s_numerical

        self.n_sim = len([f for f in os.listdir(self.dataset_path) if f.endswith('.h5')])
        self.n_data_per_sim = (n_sim_frame - self.horizon + 1) // interval

        if dataset_type == 'real' or split_numerical:
            self.n_data_train = int(self.n_sim * self.n_data_per_sim * train_ratio)

            self.n_data_val_test = self.n_sim * self.n_data_per_sim - self.n_data_train
            self.n_data_val = int(self.n_data_val_test * 0.5)
            self.n_data_test = self.n_data_val_test - self.n_data_val

            self.n_sim_in_distribution = n_sim_in_distribution
            self.n_sim_out_distribution = n_sim_out_distribution
            self.n_data_in_distribution = self.n_sim_in_distribution * self.n_data_per_sim
            self.n_data_out_distribution = self.n_sim_out_distribution * self.n_data_per_sim
            self.n_data_remain = self.n_data_val_test - self.n_data_in_distribution - self.n_data_out_distribution

        elif dataset_type == 'numerical':
            self.n_data_train = self.n_sim * self.n_data_per_sim
            self.n_data_val_test, self.n_data_val, self.n_data_test = 0, 0, 0
            self.n_sim_in_distribution, self.n_sim_out_distribution = 0, 0
            self.n_data_in_distribution, self.n_data_out_distribution = 0, 0
            self.n_data_remain = 0
            
        else:
            raise Exception(f"Dataset type {dataset_type} not supported.")

    def _get_test_mode_sim_and_time_ids(self):
        raise Exception(NotImplementedError)

    def _update_n_sim_in_out_distribution(self):
        raise Exception(NotImplementedError)
        
    def _parse_h5_files(self):
        raise Exception(NotImplementedError)

    def _separate_test_params(self):
        raise Exception(NotImplementedError)

    def _assign_sim_and_time_ids(self):
        raise Exception(NotImplementedError)

    def __getitem__(self, idx):
        raise Exception(NotImplementedError)

    def __len__(self):
        raise Exception(NotImplementedError)


def gaussian_kernel(size, sigma):
    """
    size: size of the kernel
    sigma: standard deviation of the kernel
    """
    kernel_1d = torch.linspace(-(size // 2), size // 2, size)
    kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
    
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.view(-1, 1) @ kernel_1d.view(1, -1)
    kernel_2d = kernel_2d / kernel_2d.sum()
    
    return kernel_2d

def apply_gaussian_blur(data, kernel_size, sigma):
    '''
    data: (T, H, W, C)
    '''
    kernel = gaussian_kernel(kernel_size, sigma).float().unsqueeze(0).unsqueeze(-1)
    blurred_data = F.conv2d(data, kernel, padding=kernel_size//2)

    return blurred_data