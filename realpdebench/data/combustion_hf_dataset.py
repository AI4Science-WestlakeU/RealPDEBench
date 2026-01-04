"""
Hugging Face Dataset wrapper for CombustionDataset.

This module provides a drop-in replacement for CombustionDataset that loads
pre-converted sharded Arrow files using HF Datasets library. It maintains EXACT 
same behavior as the original:
- Same sample ordering (from Arrow data)
- Same data loading logic (decoded from binary columns)
- Same channel concatenation (observed + unobserved channels)
- Same stochastic operations (mask_prob, noise_scale) when seeds are set
- Same filtering logic (test_mode, N_autoregressive)

Usage:
    from realpdebench.data.combustion_hf_dataset import CombustionHFDataset
    
    dataset = CombustionHFDataset(
        dataset_name="combustion",
        dataset_root="/wutailin/real_benchmark/",
        dataset_type="real",
        mode="train",
    )
    
    input_tensor, output_tensor = dataset[0]

"""

import os
import random
import logging
import json
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

from realpdebench.data.dataset import RealDataset, apply_gaussian_blur


# Number of unobserved/numerical channels for combustion dataset
NUMERICAL_CHANNEL = 15


class CombustionHFDataset(RealDataset):
    """
    HF Arrow-backed dataset for combustion data.
    
    Inherits from RealDataset for interface compatibility but replaces
    HDF5 loading with HF Arrow loading.
    """
    
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        dataset_type: str,
        mode: str,
        hf_auto_download: bool = False,
        hf_repo_id: str = "AI4Science-WestlakeU/RealPDEBench",
        hf_endpoint: str | None = None,
        hf_revision: str | None = None,
        test_mode: str = "all",
        mask_prob: float = 0.8,
        in_step: int = 20,
        out_step: int = 20,
        N_autoregressive: int = 1,
        interval: int = 1,
        train_ratio: float = 0.5,
        split_numerical: bool = False,
        trunk_length: int = 101,
        noise_scale: float = 0.0,
        n_sim_in_distribution: int = 0,
        n_sim_out_distribution: int = 0,
        n_sim_frame: int = 2001,
        sub_s_real: int = 2,
        sub_s_numerical: int = 2,
        noise_type: str = "gaussian",
        optical_kernel_size: int = 4,
        optical_sigma: float = 1.0,
    ):
        Dataset.__init__(self)
        
        # Store dataset identification
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.dataset_type = dataset_type
        self.mode = mode
        self.test_mode = test_mode
        
        # Paths
        self.dataset_dir = os.path.join(dataset_root, dataset_name)
        self.dataset_path = os.path.join(self.dataset_dir, dataset_type)
        self.surrogate_dataset_path = os.path.join(self.dataset_dir, "surrogate")
        self.hf_dataset_dir = os.path.join(self.dataset_dir, "hf_dataset")

        # Time step configuration
        self.in_step = in_step
        self.out_step = out_step * N_autoregressive
        self.N_autoregressive = N_autoregressive
        self.interval = interval
        self.horizon = self.in_step + self.out_step
        self.n_sim_frame = n_sim_frame
        self.trunk_length = trunk_length
        
        # Spatial subsampling (NOTE: already applied during Arrow conversion)
        self.sub_s_real = sub_s_real
        self.sub_s_numerical = sub_s_numerical
        self.sub_s = sub_s_real if dataset_type == "real" else sub_s_numerical
        
        # Stochastic parameters
        self.mask_prob = mask_prob
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        self.optical_kernel_size = optical_kernel_size
        self.optical_sigma = optical_sigma
        
        # Combustion-specific
        self.numerical_channel = NUMERICAL_CHANNEL
        
        # Load sharded Arrow dataset via HF Dataset
        arrow_path = os.path.join(self.hf_dataset_dir, f"{dataset_type}_{mode}")
        if not os.path.exists(arrow_path):
            need_test_params_json = mode in ["val", "test"] and test_mode != "all"
            from realpdebench.hf_download import ensure_hf_artifacts

            ensure_hf_artifacts(
                dataset_root=dataset_root,
                scenario=dataset_name,
                dataset_type=dataset_type,
                split=mode,
                need_test_params_json=need_test_params_json,
                hf_auto_download=hf_auto_download,
                repo_id=hf_repo_id,
                endpoint=hf_endpoint,
                revision=hf_revision,
            )

        if not os.path.exists(arrow_path):
            what = "all" if (mode in ["val", "test"] and test_mode != "all") else "hf_dataset"
            raise FileNotFoundError(
                f"HF Arrow dataset not found: {arrow_path}\n"
                "To download from HF Hub, run (example):\n"
                f"  realpdebench download --dataset-root {dataset_root} --scenario {dataset_name} "
                f"--what {what} --dataset-type {dataset_type} --split {mode}\n"
                "Or generate it locally from HDF5 via `python -m realpdebench.utils.convert_hdf5_to_hf ...`."
            )
        
        logging.info(f"Loading HF Dataset from Arrow: {arrow_path}")
        self.hf_dataset = load_from_disk(arrow_path)
        
        # Load test params for filtering (needed for test_mode)
        self.in_dist_test_params = None
        self.out_dist_test_params = None
        self.remain_params = None
        
        if mode in ["val", "test"] and test_mode != "all":
            self._load_test_params()
        
        # Build valid indices list
        self._valid_indices = list(range(len(self.hf_dataset)))
        
        # Get sim_ids and time_ids for filtering
        self._sim_ids = self.hf_dataset["sim_id"]
        self._time_ids = self.hf_dataset["time_id"]
        
        # Apply test_mode filtering
        if mode in ["val", "test"] and test_mode != "all":
            self._apply_test_mode_filter()
        
        # Apply N_autoregressive filtering
        if N_autoregressive > 1:
            self._apply_autoregressive_filter()
        
        logging.info(f"CombustionHFDataset: mode={mode}, type={dataset_type}, test_mode={test_mode}")
        logging.info(f"CombustionHFDataset: {len(self._valid_indices)} samples")
    
    def _load_test_params(self):
        """Load test parameter files for filtering (JSON)."""
        in_dist_path = os.path.join(
            self.dataset_dir, f"in_dist_test_params_{self.dataset_type}.json"
        )
        out_dist_path = os.path.join(
            self.dataset_dir, f"out_dist_test_params_{self.dataset_type}.json"
        )
        remain_path = os.path.join(
            self.dataset_dir, f"remain_params_{self.dataset_type}.json"
        )
        
        for p in [in_dist_path, out_dist_path, remain_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Missing JSON test params file: {p}\n"
                    "Run `python realpdebench/utils/convert_hdf5_to_hf.py --export_test_params_json ...` "
                    "to generate JSON metadata."
                )
        
        with open(in_dist_path, "r") as f:
            self.in_dist_test_params = json.load(f)
        with open(out_dist_path, "r") as f:
            self.out_dist_test_params = json.load(f)
        with open(remain_path, "r") as f:
            self.remain_params = json.load(f)
        
        if not isinstance(self.in_dist_test_params, dict) or not isinstance(self.out_dist_test_params, dict) or not isinstance(self.remain_params, dict):
            raise TypeError("Test params JSON files must contain dict objects.")
    
    def _apply_test_mode_filter(self):
        """Filter samples based on test_mode (in_dist, out_dist, seen, unseen)."""
        if self.test_mode == "in_dist":
            target_sim_ids = set(self.in_dist_test_params.keys())
        elif self.test_mode == "out_dist":
            target_sim_ids = set(self.out_dist_test_params.keys())
        elif self.test_mode == "seen":
            target_sim_ids = set(self.remain_params.keys())
        elif self.test_mode == "unseen":
            target_sim_ids = (
                set(self.in_dist_test_params.keys()) | 
                set(self.out_dist_test_params.keys())
            )
        else:
            raise ValueError(f"Invalid test_mode: {self.test_mode}")
        
        self._valid_indices = [
            i for i in self._valid_indices
            if self._sim_ids[i] in target_sim_ids
        ]
        logging.info(f"    After test_mode filter ({self.test_mode}): {len(self._valid_indices)} samples")
    
    def _apply_autoregressive_filter(self):
        """Filter out samples that would exceed frame limit for autoregressive."""
        self._valid_indices = [
            i for i in self._valid_indices
            if self._time_ids[i] + self.horizon < self.n_sim_frame
        ]
        logging.info(f"    After autoregressive filter: {len(self._valid_indices)} samples")
    
    def _decode_array(
        self, 
        binary_data: bytes, 
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """Decode binary data to numpy array."""
        return np.frombuffer(binary_data, dtype=dtype).reshape(shape)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a single sample from HF Dataset (Arrow-backed).
        
        Returns:
            (input, output) tuple of torch tensors with shape (T, H, W, C)
            where C = 1 + numerical_channel (= 16 total)
        """
        # Map to actual dataset index
        dataset_idx = self._valid_indices[idx]
        row = self.hf_dataset[dataset_idx]
        
        shape = (row["shape_t"], row["shape_h"], row["shape_w"])
        
        if self.dataset_type == "real":
            # Real data: load observed, append zeros for numerical channels
            observed = self._decode_array(row["observed"], shape)
            # Add channel dimension
            data = torch.tensor(observed, dtype=torch.float32).unsqueeze(-1)
            
            # Append zeros for unobserved channels
            unobserved = torch.zeros(
                data.shape[0], data.shape[1], data.shape[2], self.numerical_channel
            )
            data = torch.cat([data, unobserved], dim=-1)
        else:
            # Numerical data: load observed (surrogate) + numerical channels
            observed = self._decode_array(row["observed"], shape)
            surrogate_data = torch.tensor(observed, dtype=torch.float32).unsqueeze(-1)
            
            # Apply mask_prob: randomly decide whether to use zeros or actual numerical data
            if random.random() < self.mask_prob:
                numerical = torch.zeros(
                    surrogate_data.shape[0], surrogate_data.shape[1],
                    surrogate_data.shape[2], self.numerical_channel
                )
            else:
                # Load actual numerical data
                num_channels = row["numerical_channels"]
                numerical = self._decode_array(
                    row["numerical"],
                    (*shape, num_channels)
                )
                numerical = torch.tensor(numerical, dtype=torch.float32)
            
            data = torch.cat([surrogate_data, numerical], dim=-1)
        
        input_data = data[:self.in_step]
        output_data = data[self.in_step:]
        
        # Apply noise augmentation (only for numerical data)
        if self.noise_scale > 0 and self.dataset_type == "numerical":
            if self.noise_type == "gaussian":
                input_data = input_data + input_data * torch.randn_like(input_data) * self.noise_scale
                output_data = output_data + output_data * torch.randn_like(output_data) * self.noise_scale
            elif self.noise_type == "poisson":
                input_data = input_data + torch.poisson(input_data) * self.noise_scale
                output_data = output_data + torch.poisson(output_data) * self.noise_scale
            elif self.noise_type == "optical":
                input_data = apply_gaussian_blur(
                    input_data, self.optical_kernel_size, self.optical_sigma
                )
                output_data = apply_gaussian_blur(
                    output_data, self.optical_kernel_size, self.optical_sigma
                )
            else:
                raise ValueError(f"Invalid noise type: {self.noise_type}")
        
        return input_data, output_data  # T, H, W, C
    
    def __len__(self) -> int:
        return len(self._valid_indices)
