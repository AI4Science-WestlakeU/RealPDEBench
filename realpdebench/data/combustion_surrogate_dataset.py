import torch
import numpy as np
import os
import h5py
import random
import re
import logging
from torch.utils.data import Dataset



class SurrogateDataset(Dataset):
    def __init__(self, 
            dataset_name,
            dataset_root, 
            mode, 
            train_ratio=0.8,
            step=20,
            n_sim_frame=40,
            n_sim_frame_test=2001,
            sub_s_real=1,
            sub_s_numerical=1,
            ):
        super().__init__()
        assert dataset_name == "combustion"

        self.dataset_dir = os.path.join(dataset_root, dataset_name)
        self.real_dataset_path = os.path.join(self.dataset_dir, "real_surrogate_train")
        self.numerical_dataset_path = os.path.join(self.dataset_dir, "numerical_surrogate_train")
        self.time_ids = list(range(n_sim_frame-step))

        self.sim_ids = [f for f in os.listdir(self.numerical_dataset_path) if f.endswith('.h5')]
        self.n_sim = len(self.sim_ids)

        self.mode = mode
        self.step = step
        self.n_sim_frame = n_sim_frame
        self.n_sim_frame_test = n_sim_frame_test
        self.train_ratio = train_ratio
        self.sub_s_real = sub_s_real
        self.sub_s_numerical = sub_s_numerical
        self.numerical_channel = 15

    def __getitem__(self, idx):
        sim_id = random.choice(self.sim_ids)
        time_id = random.choice(self.time_ids)
        match = re.match(r'(\d+)NH3_(\d+\.?\d*)\.h5', sim_id)
        gas_ratio = int(match.group(1)) 
        equivalence_ratio = float(match.group(2))

        with h5py.File(os.path.join(self.real_dataset_path, f"{sim_id}"), "r") as f:
            real_data = f[f"trajectory"][time_id:time_id+self.step, ::self.sub_s_real, ::self.sub_s_real]
            real_data = torch.tensor(real_data, dtype=torch.float).unsqueeze(-1)
        
        with h5py.File(os.path.join(self.numerical_dataset_path, f"{sim_id}"), "r") as f:
            numerical_data = f["measured_data"][time_id:time_id+self.step, 
                                ::self.sub_s_numerical, ::self.sub_s_numerical] #, [1,8]]
            numerical_data = torch.tensor(numerical_data, dtype=torch.float)

            gas_ratio_channel = torch.ones_like(numerical_data[..., [0]]) * gas_ratio
            equivalence_ratio_channel = torch.ones_like(numerical_data[..., [0]]) * equivalence_ratio
            numerical_data = torch.cat([numerical_data, gas_ratio_channel, equivalence_ratio_channel], dim=-1)
        
        input = numerical_data
        output = real_data
        
        return input, output # T, S, S, C

    def __len__(self):
        if self.mode == 'train':
            return int(self.n_sim * self.n_sim_frame)
        else:
            return int(self.n_sim * self.n_sim_frame / self.train_ratio * (1 - self.train_ratio))


if __name__ == "__main__":
    dataset = SurrogateDataset(
        dataset_name="combustion",
        dataset_root="/wutailin/real_benchmark/",
        mode="train",
        )
    
    print(len(dataset))
    print(dataset[0][0].shape, dataset[0][1].shape)

    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    # for input, output in dataloader:
    #     print(input.shape, output.shape)
    #     break