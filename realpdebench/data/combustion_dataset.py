import torch
import numpy as np
import os
import h5py
import re
import random
import logging
from realpdebench.data.dataset import RealDataset
from realpdebench.data.dataset import apply_gaussian_blur



class CombustionDataset(RealDataset):
    def __init__(self, 
            dataset_name, 
            dataset_root, 
            dataset_type,
            mode, 
            test_mode="all", 
            mask_prob=0.8,
            in_step=20,
            out_step=20, 
            N_autoregressive=1,
            interval=1,
            train_ratio=0.5,
            split_numerical=False,
            trunk_length=101,
            noise_scale=0.,
            n_sim_in_distribution=0,
            n_sim_out_distribution=0,
            n_sim_frame=2001,
            sub_s_real=2,
            sub_s_numerical=2,
            noise_type='gaussian',
            optical_kernel_size=4,
            optical_sigma=1.0
            ):
        super().__init__(dataset_name, dataset_root, dataset_type, mode, test_mode, mask_prob, in_step, out_step, 
                        N_autoregressive, interval, train_ratio, split_numerical, trunk_length, noise_scale, 
                        n_sim_in_distribution, n_sim_out_distribution, n_sim_frame, sub_s_real, sub_s_numerical)

        self.numerical_channel = 15
        self.surrogate_dataset_path = os.path.join(self.dataset_dir, 'surrogate')

        in_dist_test_params = os.path.join(self.dataset_dir, f"in_dist_test_params_{self.dataset_type}.pt")
        out_dist_test_params = os.path.join(self.dataset_dir, f"out_dist_test_params_{self.dataset_type}.pt")
        remain_params = os.path.join(self.dataset_dir, f"remain_params_{self.dataset_type}.pt")
        sim_id_mapping = os.path.join(self.dataset_dir, f"sim_id_mapping_{self.dataset_type}.pt")
        time_id_mapping = os.path.join(self.dataset_dir, f"time_id_mapping_{self.dataset_type}.pt")

        logging.info(f'Path of dataset ids: {in_dist_test_params}, {out_dist_test_params}, {remain_params}, '+\
                    f'{sim_id_mapping}, {time_id_mapping}')

        try:
            self.in_dist_test_params = torch.load(in_dist_test_params)
            self.out_dist_test_params = torch.load(out_dist_test_params)
            self.remain_params = torch.load(remain_params)
            self.sim_id_mapping = torch.load(sim_id_mapping)
            self.time_id_mapping = torch.load(time_id_mapping)

        except:
            raise ValueError("Error loading dataset ids")
            self.file_params = self._parse_h5_files()
            self.in_dist_test_params, self.out_dist_test_params, self.remain_params \
                                                        = self._separate_test_params()
            self._update_n_sim_in_out_distribution()
            self.sim_id_mapping, self.time_id_mapping = self._assign_sim_and_time_ids()

            torch.save(self.in_dist_test_params, in_dist_test_params)
            torch.save(self.out_dist_test_params, out_dist_test_params)
            torch.save(self.remain_params, remain_params)
            torch.save(self.sim_id_mapping, sim_id_mapping)
            torch.save(self.time_id_mapping, time_id_mapping)

            logging.info(f"in_dist_test_params: {self.in_dist_test_params.keys()}")
            logging.info(f"out_dist_test_params: {self.out_dist_test_params.keys()}")
            logging.info(f"remain_params: {self.remain_params.keys()}")
        
        if self.mode == 'val' or self.mode == 'test':
            if self.test_mode != "all":
                self._get_test_mode_sim_and_time_ids()
            if self.N_autoregressive > 1:
                self._filter_time_ids()
    
        logging.info(f"train: {len(self.sim_id_mapping['train'])}")
        logging.info(f"val: {len(self.sim_id_mapping['val'])}")
        logging.info(f"test: {len(self.sim_id_mapping['test'])}")
        
    def _get_test_mode_sim_and_time_ids(self):
        '''
        Get val and test sim_ids and time_ids according to test mode.
        '''
        test_sim_ids = self.sim_id_mapping[self.mode]
        test_time_ids = self.time_id_mapping[self.mode]

        if self.test_mode == "in_dist":
            target_sim_id = self.in_dist_test_params.keys()
        elif self.test_mode == "out_dist":
            target_sim_id = self.out_dist_test_params.keys()
        elif self.test_mode == "seen":
            target_sim_id = self.remain_params.keys()
        elif self.test_mode == "unseen":
            target_sim_id = list(self.in_dist_test_params.keys()) + list(self.out_dist_test_params.keys())
        else:
            raise ValueError(f"Invalid test_mode: {self.test_mode}")
        
        filtered_sim_ids = []
        filtered_time_ids = []
        for sim_id, time_id in zip(test_sim_ids, test_time_ids):
            if sim_id in target_sim_id:
                filtered_sim_ids.append(sim_id)
                filtered_time_ids.append(time_id)
        self.sim_id_mapping[self.mode] = filtered_sim_ids
        self.time_id_mapping[self.mode] = filtered_time_ids

    def _filter_time_ids(self):
        filtered_sim_ids = []
        filtered_time_ids = []
        for sim_id, time_id in zip(self.sim_id_mapping[self.mode], self.time_id_mapping[self.mode]):
            if time_id + self.horizon < self.n_sim_frame:
                filtered_sim_ids.append(sim_id)
                filtered_time_ids.append(time_id)
        self.sim_id_mapping[self.mode] = filtered_sim_ids
        self.time_id_mapping[self.mode] = filtered_time_ids
        
    def _parse_h5_files(self):
        """
        Parse .h5 files in dataset_path and extract gas ratio and equivalence ratio parameters.
        
        Returns:
            dict: Dictionary with filename as key and (gas_ratio, equivalence_ratio) as value
                  Example: {'40NH3_1.1.h5': (40, 1.1)}
        """
        file_params = {}
        
        if not os.path.exists(self.dataset_path):
            print(f"Warning: Dataset path {self.dataset_path} does not exist")
            return file_params
        
        for filename in os.listdir(self.dataset_path):
            if filename.endswith('.h5'):
                # Parse filename using regex pattern xxNH3_yy.h5
                match = re.match(r'(\d+)NH3_(\d+\.?\d*)\.h5', filename)
                gas_ratio = int(match.group(1)) 
                equivalence_ratio = float(match.group(2))
                file_params[filename] = (gas_ratio, equivalence_ratio)
        
        return file_params

    def _separate_test_params(self):
        """
        Separate test parameter pairs into in-distribution (middle values) and out-of-distribution 
        (edge values).
        
        Returns:
            tuple: (in_dist_test_params, out_dist_test_params, remain_params)
                   Each is a dict with filename as key and (gas_ratio, equivalence_ratio) as value
                   Example: {'40NH3_1.1.h5': (40, 1.1)}
        """
        all_params = [(filename, gas_ratio, equiv_ratio) 
                     for filename, (gas_ratio, equiv_ratio) in self.file_params.items()]
        
        all_params.sort(key=lambda x: (x[1], x[2]))
        
        gas_ratios = [params[1] for params in all_params]
        equiv_ratios = [params[2] for params in all_params]
        
        min_gas_ratio = min(gas_ratios)
        max_gas_ratio = max(gas_ratios)
        min_equiv_ratio = min(equiv_ratios)
        max_equiv_ratio = max(equiv_ratios)
        
        # out-of-distribution test params
        out_dist_test_params = []
        for filename, gas_ratio, equiv_ratio in all_params:
            if (gas_ratio == min_gas_ratio or gas_ratio == max_gas_ratio or 
                equiv_ratio == min_equiv_ratio or equiv_ratio == max_equiv_ratio):
                out_dist_test_params.append((filename, gas_ratio, equiv_ratio))
        
        # in-distribution test params
        remain_params = [params for params in all_params 
                          if params not in out_dist_test_params]
        
        gas_ratio_groups = {}
        for filename, gas_ratio, equiv_ratio in remain_params:
            if gas_ratio not in gas_ratio_groups:
                gas_ratio_groups[gas_ratio] = []
            gas_ratio_groups[gas_ratio].append((filename, gas_ratio, equiv_ratio))
        
        group_edge_params = []
        for gas_ratio, group_params in gas_ratio_groups.items():
            group_equiv_ratios = [params[2] for params in group_params]
            min_group_equiv = min(group_equiv_ratios)
            max_group_equiv = max(group_equiv_ratios)
                
            for filename, gr, equiv_ratio in group_params:
                if equiv_ratio == min_group_equiv or equiv_ratio == max_group_equiv:
                    group_edge_params.append((filename, gr, equiv_ratio))
        
        in_dist_test_params = [params for params in remain_params 
                                if params not in group_edge_params]
        
        # random select
        random.shuffle(in_dist_test_params)
        random.shuffle(out_dist_test_params)
        in_dist_test_params = in_dist_test_params[:self.n_sim_in_distribution]
        out_dist_test_params = out_dist_test_params[:self.n_sim_out_distribution]
        
        in_dist_test_params_dict = {filename: (gas_ratio, equiv_ratio) 
                                     for filename, gas_ratio, equiv_ratio in in_dist_test_params}
        out_dist_test_params_dict = {filename: (gas_ratio, equiv_ratio) 
                                      for filename, gas_ratio, equiv_ratio in out_dist_test_params}

        # remain params
        remain_params = [params for params in all_params 
                          if params not in in_dist_test_params and params not in out_dist_test_params]
        remain_params_dict = {filename: (gas_ratio, equiv_ratio) 
                              for filename, gas_ratio, equiv_ratio in remain_params}
        
        return in_dist_test_params_dict, out_dist_test_params_dict, remain_params_dict

    def _update_n_sim_in_out_distribution(self):
        self.n_sim_in_distribution = len(self.in_dist_test_params.keys())
        self.n_sim_out_distribution = len(self.out_dist_test_params.keys())

        self.n_data_in_distribution = self.n_sim_in_distribution * self.n_data_per_sim
        self.n_data_out_distribution = self.n_sim_out_distribution * self.n_data_per_sim
        self.n_data_remain = self.n_data_val_test - self.n_data_in_distribution - self.n_data_out_distribution

    def _assign_sim_and_time_ids(self):
        """
        Assign sim_ids and time_ids based on parameter distributions.
        - train: all from remain_params
        - val: 1/3 from in_dist, 1/3 from out_dist, 1/3 from remain_params
        - test: 1/3 from in_dist, 1/3 from out_dist, 1/3 from remain_params
        train and test&val time ids are from different trunks
        
        Returns:
            tuple: (sim_id_mapping, time_id_mapping)
        """
        in_dist_pairs = []
        for sim_id in self.in_dist_test_params.keys():
            for time_id in range(self.n_data_per_sim):
                time_id = time_id * self.interval
                in_dist_pairs.append((sim_id, time_id))

        out_dist_pairs = []
        for sim_id in self.out_dist_test_params.keys():
            for time_id in range(self.n_data_per_sim):
                time_id = time_id * self.interval
                out_dist_pairs.append((sim_id, time_id))

        remain_pairs = []
        for sim_id in self.remain_params.keys():
            for time_id in range(self.n_data_per_sim):
                time_id = time_id * self.interval
                remain_pairs.append((sim_id, time_id))
        
        n_trunks = int(np.ceil(self.n_sim_frame / self.trunk_length))
        trunks = []
        for sim_id in self.remain_params.keys():
            for trunk_idx in range(n_trunks):
                start_time_idx = trunk_idx * self.trunk_length
                end_time_idx = min((trunk_idx + 1) * self.trunk_length, self.n_sim_frame - self.horizon + 1)
                
                trunk_pairs = []
                for time_id in range(start_time_idx, end_time_idx, self.interval):
                    trunk_pairs.append((sim_id, time_id))
                trunks.append(trunk_pairs)
                
        random.shuffle(trunks)
        remain_pairs_train = []
        remain_pairs_valtest = []
        for trunk in trunks:
            if len(remain_pairs_train) < self.n_data_train:
                remain_pairs_train.extend(trunk)
            else:
                remain_pairs_valtest.extend(trunk)
        
        random.shuffle(in_dist_pairs)
        random.shuffle(out_dist_pairs)
        random.shuffle(remain_pairs_train)
        random.shuffle(remain_pairs_valtest)
        
        train_pairs = remain_pairs_train
        
        val_in_dist_pairs = in_dist_pairs[:self.n_data_in_distribution // 2]
        val_out_dist_pairs = out_dist_pairs[:self.n_data_out_distribution // 2]
        val_remain_pairs = remain_pairs_valtest[:len(remain_pairs_valtest) // 2]

        test_in_dist_pairs = in_dist_pairs[self.n_data_in_distribution // 2:]
        test_out_dist_pairs = out_dist_pairs[self.n_data_out_distribution // 2:]
        test_remain_pairs = remain_pairs_valtest[len(remain_pairs_valtest) // 2:]

        val_pairs = val_in_dist_pairs + val_out_dist_pairs + val_remain_pairs
        test_pairs = test_in_dist_pairs + test_out_dist_pairs + test_remain_pairs
        
        random.shuffle(val_pairs)
        random.shuffle(test_pairs)
        random.shuffle(train_pairs)
        
        sim_id_mapping = {
            'train': [sim_id for sim_id, _ in train_pairs],
            'val': [sim_id for sim_id, _ in val_pairs],
            'test': [sim_id for sim_id, _ in test_pairs]
        }
        
        time_id_mapping = {
            'train': [time_id for _, time_id in train_pairs],
            'val': [time_id for _, time_id in val_pairs],
            'test': [time_id for _, time_id in test_pairs]
        }
        
        return sim_id_mapping, time_id_mapping

    def __getitem__(self, idx):

        sim_id = self.sim_id_mapping[self.mode][idx]
        time_id = self.time_id_mapping[self.mode][idx]

        if self.dataset_type == "real":
            with h5py.File(os.path.join(self.dataset_path, f"{sim_id}"), "r") as f:
                data = f[f"trajectory"][time_id:time_id + self.horizon, 
                                        ::self.sub_s_real, ::self.sub_s_real]
                data = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)

                unobserved_data = torch.zeros(data.shape[0], data.shape[1], data.shape[2], self.numerical_channel)

            data = torch.cat([data, unobserved_data], dim=-1)

        else:
            with h5py.File(os.path.join(self.surrogate_dataset_path, f"{sim_id}"), "r") as f:
                surrogate_data = f["measured_data"][time_id:time_id + self.horizon, 
                                ::self.sub_s_numerical, ::self.sub_s_numerical]
                surrogate_data = torch.tensor(surrogate_data, dtype=torch.float32).unsqueeze(-1)
    
            if random.random() < self.mask_prob:
                numerical_data = torch.zeros(surrogate_data.shape[0], surrogate_data.shape[1], 
                                    surrogate_data.shape[2], self.numerical_channel)
            else:
                with h5py.File(os.path.join(self.dataset_path, f"{sim_id}"), "r") as f:
                    numerical_data = f["measured_data"][time_id:time_id + self.horizon, 
                                    ::self.sub_s_numerical, ::self.sub_s_numerical]
                    numerical_data = torch.tensor(numerical_data, dtype=torch.float32)
            data = torch.cat([surrogate_data, numerical_data], dim=-1)


        input = data[:self.in_step]
        output = data[self.in_step:]
        
        if self.noise_scale > 0 and self.dataset_type == "numerical":
            if self.noise_type == "gaussian":
                input += input * torch.randn_like(input) * self.noise_scale
                output += output * torch.randn_like(output) * self.noise_scale
            elif self.noise_type == "poisson":
                input += torch.poisson(input) * self.noise_scale
                output += torch.poisson(output) * self.noise_scale
            elif self.noise_type == "optical":
                input = apply_gaussian_blur(input, self.optical_kernel_size, self.optical_sigma)
                output = apply_gaussian_blur(output, self.optical_kernel_size, self.optical_sigma)
            else:
                raise ValueError(f"Invalid noise type: {self.noise_type}")

        return input, output # T, S, S, C

    def __len__(self):
        return len(self.sim_id_mapping[self.mode])


if __name__ == "__main__":
    dataset = CombustionDataset(
        dataset_name="combustion",
        dataset_root="/wutailin/real_benchmark/",
        dataset_type="real",
        mode="train",
        # test_mode="out_dist",
        )
    
    print(len(dataset))
    print(dataset[0][0].shape, dataset[0][1].shape)

    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    # for input, output in dataloader:
    #     print(input.shape, output.shape)
    #     break