"""
Hugging Face Dataset wrappers for Fluid datasets - V2 (Lazy Slicing).

This module provides drop-in replacements for all fluid datasets that load
pre-converted Arrow files containing COMPLETE trajectories and perform
DYNAMIC slicing at runtime based on N_autoregressive.

Key differences from V1:
- Trajectories stored as complete time series (no pre-slicing)
- Index files (JSON) map (sim_id, time_id) for train/val/test splits
- Dynamic slicing: data[time_id : time_id + horizon] at runtime
- Supports any N_autoregressive value without re-conversion

Classes:
    - FluidHFDataset: Base class with lazy slicing logic
    - CylinderHFDataset: Wrapper for Cylinder
    - FSIHFDataset: Wrapper for FSI
    - ControlledCylinderHFDataset: Wrapper for ControlledCylinder
    - FoilHFDataset: Wrapper for Foil

Each class maintains EXACT same output behavior as the original H5 datasets:
- Same sample ordering (from index files)
- Same input/output shapes
- Same channel structure (u, v, p) with mask_prob for pressure
- Same filtering logic (test_mode, N_autoregressive)
"""

import os
import re
import json
import random
import logging
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

from realpdebench.data.dataset import RealDataset, apply_gaussian_blur


class FluidHFDataset(RealDataset):
    """
    Base class for fluid HF Arrow-backed datasets with lazy slicing.
    
    Uses V2 format:
    - Trajectory data: {hf_dataset_dir}/{dataset_type}/ (Arrow)
    - Index files: {hf_dataset_dir}/{split}_index_{dataset_type}.json
    
    Subclasses should set class attributes:
    - file_name_pattern: Regex pattern for parsing simulation parameters
    - condition_on_para: Whether to append parameters as input channels
    - d, center_x, center_y: Cylinder geometry parameters
    """
    
    # Subclasses should override these
    file_name_pattern: str = r"(\d+)\.h5"
    condition_on_para: bool = False
    d: int = 18  # cylinder diameter
    center_x: int = 32  # center x coordinate
    center_y: int = 32  # center y coordinate
    
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
        mask_prob: float = 0.5,
        in_step: int = 20,
        out_step: int = 20,
        N_autoregressive: int = 1,
        interval: int = 10,
        train_ratio: float = 0.5,
        split_numerical: bool = False,
        trunk_length: int = 190,
        noise_scale: float = 0.0,
        n_sim_in_distribution: int = 0,
        n_sim_out_distribution: int = 4,
        n_sim_frame: int = 3990,
        sub_s_real: int = 1,
        sub_s_numerical: int = 2,
        noise_type: str = "gaussian",
        optical_kernel_size: int = 4,
        optical_sigma: float = 1.0,
    ):
        # Skip RealDataset.__init__() - we don't need HDF5 file counting
        Dataset.__init__(self)

        # Check data version compatibility
        from realpdebench import check_data_version
        check_data_version(dataset_root)

        # Store dataset identification
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.dataset_type = dataset_type
        self.mode = mode
        self.test_mode = test_mode
        
        # Time step configuration (matches H5 logic exactly)
        self.in_step = in_step
        self.out_step = out_step * N_autoregressive  # Key: multiply by N_autoregressive
        self.N_autoregressive = N_autoregressive
        self.interval = interval
        self.horizon = self.in_step + self.out_step
        self.n_sim_frame = n_sim_frame
        self.trunk_length = trunk_length
        
        # Spatial subsampling (applied at runtime, stored at full resolution)
        self.sub_s_real = sub_s_real
        self.sub_s_numerical = sub_s_numerical
        self.sub_s = sub_s_real if dataset_type == "real" else sub_s_numerical
        
        # Stochastic parameters
        self.mask_prob = mask_prob
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        self.optical_kernel_size = optical_kernel_size
        self.optical_sigma = optical_sigma
        
        # Paths
        self.dataset_dir = os.path.join(dataset_root, dataset_name)
        self.dataset_path = os.path.join(self.dataset_dir, dataset_type)
        self.hf_dataset_dir = os.path.join(self.dataset_dir, "hf_dataset")
        
        # Load trajectory data (all sim_ids for this dataset_type)
        trajectory_path = os.path.join(self.hf_dataset_dir, dataset_type)
        index_path = os.path.join(self.hf_dataset_dir, f"{mode}_index_{dataset_type}.json")
        if not (os.path.exists(trajectory_path) and os.path.exists(index_path)):
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

        if not os.path.exists(trajectory_path):
            raise FileNotFoundError(
                f"HF Arrow trajectories not found: {trajectory_path}\n"
                "Run `python -m realpdebench.utils.convert_hdf5_to_hf ...` to generate V2 format."
            )
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Index file not found: {index_path}\n"
                "Run `python -m realpdebench.utils.convert_hdf5_to_hf ...` to generate V2 format."
            )

        logging.info(f"Loading trajectories from: {trajectory_path}")
        self.trajectories = load_from_disk(trajectory_path)

        # Build sim_id -> trajectory_idx mapping
        self._sim_id_to_idx: Dict[str, int] = {}
        for i in range(len(self.trajectories)):
            sim_id = self.trajectories[i]["sim_id"]
            self._sim_id_to_idx[sim_id] = i

        logging.info(f"  Loaded {len(self.trajectories)} trajectories")

        # Load index file
        with open(index_path, "r") as f:
            self._indices: List[Dict] = json.load(f)

        logging.info(f"  Loaded {len(self._indices)} indices from {index_path}")

        # Load test params for filtering (if needed)
        self.in_dist_test_params = None
        self.out_dist_test_params = None
        self.remain_params = None
        
        if mode in ["val", "test"] and test_mode != "all":
            self._load_test_params()
            self._apply_test_mode_filter()
        
        # Apply N_autoregressive filtering (filter samples that would exceed frame limit)
        if mode in ["val", "test"] and N_autoregressive > 1:
            self._apply_autoregressive_filter()
        
        logging.info(f"{self.__class__.__name__}: mode={mode}, type={dataset_type}, test_mode={test_mode}")
        logging.info(f"{self.__class__.__name__}: {len(self._indices)} samples, horizon={self.horizon}")
    
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
                    "Generate JSON metadata from PT files."
                )
        
        with open(in_dist_path, "r") as f:
            self.in_dist_test_params = json.load(f)
        with open(out_dist_path, "r") as f:
            self.out_dist_test_params = json.load(f)
        with open(remain_path, "r") as f:
            self.remain_params = json.load(f)
    
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
        
        original_len = len(self._indices)
        self._indices = [
            entry for entry in self._indices
            if entry["sim_id"] in target_sim_ids
        ]
        logging.info(f"    After test_mode filter ({self.test_mode}): {len(self._indices)}/{original_len} samples")
    
    def _apply_autoregressive_filter(self):
        """Filter out samples that would exceed frame limit for autoregressive."""
        original_len = len(self._indices)
        self._indices = [
            entry for entry in self._indices
            if entry["time_id"] + self.horizon < self.n_sim_frame
        ]
        logging.info(f"    After autoregressive filter: {len(self._indices)}/{original_len} samples")
    
    def _decode_array(self, binary_data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Decode binary data to numpy array."""
        return np.frombuffer(binary_data, dtype=np.float32).reshape(shape)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a single sample with DYNAMIC slicing.
        
        This is the key difference from V1: we slice at runtime based on
        time_id and horizon (which depends on N_autoregressive).
        
        Returns:
            (input, output) tuple of torch tensors with shape (T, H, W, C)
        """
        # Get (sim_id, time_id) from index
        entry = self._indices[idx]
        sim_id = entry["sim_id"]
        time_id = entry["time_id"]
        
        # Look up trajectory
        traj_idx = self._sim_id_to_idx[sim_id]
        row = self.trajectories[traj_idx]
        
        # Decode complete trajectory (stored at full resolution)
        full_shape = (row["shape_t"], row["shape_h"], row["shape_w"])
        u_full = self._decode_array(row["u"], full_shape)
        v_full = self._decode_array(row["v"], full_shape)

        # DYNAMIC slicing + RUNTIME SUBSAMPLING
        # Time slicing: [time_id : time_id + horizon]
        # Spatial subsampling: [::sub_s, ::sub_s] (same as H5 loader)
        sub_s = self.sub_s
        u = u_full[time_id : time_id + self.horizon, ::sub_s, ::sub_s]
        v = v_full[time_id : time_id + self.horizon, ::sub_s, ::sub_s]

        # Handle pressure channel
        if self.dataset_type == "real":
            # Real data: p is zeros
            p = np.zeros_like(u)
        else:
            # Numerical data: apply mask_prob
            if random.random() < self.mask_prob:
                p = np.zeros_like(u)
            else:
                p_full = self._decode_array(row["p"], full_shape)
                p = p_full[time_id : time_id + self.horizon, ::sub_s, ::sub_s]
        
        # Stack channels: (T, H, W, 3)
        data = np.stack([u, v, p], axis=-1)
        
        # Split into input/output (same logic as H5)
        input_data = torch.tensor(data[:self.in_step], dtype=torch.float32)
        output_data = torch.tensor(data[self.in_step:], dtype=torch.float32)
        
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
        
        # Apply parameter conditioning (only for ControlledCylinder)
        if self.condition_on_para:
            match = re.match(self.file_name_pattern, sim_id)
            para_list = []
            for group in match.groups():
                para_data = float(group) * torch.ones_like(input_data[..., 0])
                para_list.append(para_data)
            input_data = torch.cat([input_data, torch.stack(para_list, dim=-1)], dim=-1)
        
        return input_data, output_data  # T, H, W, C
    
    def __len__(self) -> int:
        return len(self._indices)


class CylinderHFDataset(FluidHFDataset):
    """HF Dataset V2 wrapper for Cylinder (single cylinder)."""
    
    file_name_pattern = r"(\d+)\.h5"
    condition_on_para = False
    d = 18
    center_x = 32
    center_y = 32
    
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
        mask_prob: float = 0.5,
        in_step: int = 20,
        out_step: int = 20,
        N_autoregressive: int = 1,
        interval: int = 20,
        train_ratio: float = 0.5,
        split_numerical: bool = False,
        trunk_length: int = 250,
        noise_scale: float = 0.0,
        n_sim_in_distribution: int = 10,
        n_sim_out_distribution: int = 10,
        n_sim_frame: int = 3990,
        sub_s_real: int = 1,
        sub_s_numerical: int = 2,
        noise_type: str = "gaussian",
        optical_kernel_size: int = 4,
        optical_sigma: float = 1.0,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            dataset_type=dataset_type,
            mode=mode,
            hf_auto_download=hf_auto_download,
            hf_repo_id=hf_repo_id,
            hf_endpoint=hf_endpoint,
            hf_revision=hf_revision,
            test_mode=test_mode,
            mask_prob=mask_prob,
            in_step=in_step,
            out_step=out_step,
            N_autoregressive=N_autoregressive,
            interval=interval,
            train_ratio=train_ratio,
            split_numerical=split_numerical,
            trunk_length=trunk_length,
            noise_scale=noise_scale,
            n_sim_in_distribution=n_sim_in_distribution,
            n_sim_out_distribution=n_sim_out_distribution,
            n_sim_frame=n_sim_frame,
            sub_s_real=sub_s_real,
            sub_s_numerical=sub_s_numerical,
            noise_type=noise_type,
            optical_kernel_size=optical_kernel_size,
            optical_sigma=optical_sigma,
        )


class FSIHFDataset(FluidHFDataset):
    """HF Dataset V2 wrapper for FSI (double cylinder / fluid-structure interaction)."""
    
    file_name_pattern = r"(\d+)_([\d\.]+)_"
    condition_on_para = False
    d = 24
    center_x = 66
    center_y = 64
    
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
        mask_prob: float = 0.5,
        in_step: int = 20,
        out_step: int = 20,
        N_autoregressive: int = 1,
        interval: int = 10,
        train_ratio: float = 0.5,
        split_numerical: bool = False,
        trunk_length: int = 190,
        noise_scale: float = 0.0,
        n_sim_in_distribution: int = 0,
        n_sim_out_distribution: int = 4,
        n_sim_frame: int = 2173,
        sub_s_real: int = 2,
        sub_s_numerical: int = 2,
        noise_type: str = "gaussian",
        optical_kernel_size: int = 4,
        optical_sigma: float = 1.0,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            dataset_type=dataset_type,
            mode=mode,
            hf_auto_download=hf_auto_download,
            hf_repo_id=hf_repo_id,
            hf_endpoint=hf_endpoint,
            hf_revision=hf_revision,
            test_mode=test_mode,
            mask_prob=mask_prob,
            in_step=in_step,
            out_step=out_step,
            N_autoregressive=N_autoregressive,
            interval=interval,
            train_ratio=train_ratio,
            split_numerical=split_numerical,
            trunk_length=trunk_length,
            noise_scale=noise_scale,
            n_sim_in_distribution=n_sim_in_distribution,
            n_sim_out_distribution=n_sim_out_distribution,
            n_sim_frame=n_sim_frame,
            sub_s_real=sub_s_real,
            sub_s_numerical=sub_s_numerical,
            noise_type=noise_type,
            optical_kernel_size=optical_kernel_size,
            optical_sigma=optical_sigma,
        )


class ControlledCylinderHFDataset(FluidHFDataset):
    """HF Dataset V2 wrapper for ControlledCylinder with parameter conditioning."""
    
    file_name_pattern = r"(\d+)_(\d+\.?\d*)\.h5"
    condition_on_para = True
    d = 18
    center_x = 16
    center_y = 32
    
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
        mask_prob: float = 0.5,
        in_step: int = 10,
        out_step: int = 10,
        N_autoregressive: int = 1,
        interval: int = 20,
        train_ratio: float = 0.5,
        split_numerical: bool = False,
        trunk_length: int = 250,
        noise_scale: float = 0.0,
        n_sim_in_distribution: int = 10,
        n_sim_out_distribution: int = 10,
        n_sim_frame: int = 3990,
        sub_s_real: int = 1,
        sub_s_numerical: int = 2,
        noise_type: str = "gaussian",
        optical_kernel_size: int = 4,
        optical_sigma: float = 1.0,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            dataset_type=dataset_type,
            mode=mode,
            hf_auto_download=hf_auto_download,
            hf_repo_id=hf_repo_id,
            hf_endpoint=hf_endpoint,
            hf_revision=hf_revision,
            test_mode=test_mode,
            mask_prob=mask_prob,
            in_step=in_step,
            out_step=out_step,
            N_autoregressive=N_autoregressive,
            interval=interval,
            train_ratio=train_ratio,
            split_numerical=split_numerical,
            trunk_length=trunk_length,
            noise_scale=noise_scale,
            n_sim_in_distribution=n_sim_in_distribution,
            n_sim_out_distribution=n_sim_out_distribution,
            n_sim_frame=n_sim_frame,
            sub_s_real=sub_s_real,
            sub_s_numerical=sub_s_numerical,
            noise_type=noise_type,
            optical_kernel_size=optical_kernel_size,
            optical_sigma=optical_sigma,
        )


class FoilHFDataset(FluidHFDataset):
    """HF Dataset V2 wrapper for Foil."""
    
    file_name_pattern = r"(\d+)_(\d+\.?\d*)\.h5"
    condition_on_para = False
    d = 62
    center_x = 30
    center_y = 64
    
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
        mask_prob: float = 0.5,
        in_step: int = 20,
        out_step: int = 20,
        N_autoregressive: int = 1,
        interval: int = 20,
        train_ratio: float = 0.5,
        split_numerical: bool = False,
        trunk_length: int = 250,
        noise_scale: float = 0.0,
        n_sim_in_distribution: int = 10,
        n_sim_out_distribution: int = 10,
        n_sim_frame: int = 3990,
        sub_s_real: int = 2,
        sub_s_numerical: int = 2,
        noise_type: str = "gaussian",
        optical_kernel_size: int = 4,
        optical_sigma: float = 1.0,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            dataset_type=dataset_type,
            mode=mode,
            hf_auto_download=hf_auto_download,
            hf_repo_id=hf_repo_id,
            hf_endpoint=hf_endpoint,
            hf_revision=hf_revision,
            test_mode=test_mode,
            mask_prob=mask_prob,
            in_step=in_step,
            out_step=out_step,
            N_autoregressive=N_autoregressive,
            interval=interval,
            train_ratio=train_ratio,
            split_numerical=split_numerical,
            trunk_length=trunk_length,
            noise_scale=noise_scale,
            n_sim_in_distribution=n_sim_in_distribution,
            n_sim_out_distribution=n_sim_out_distribution,
            n_sim_frame=n_sim_frame,
            sub_s_real=sub_s_real,
            sub_s_numerical=sub_s_numerical,
            noise_type=noise_type,
            optical_kernel_size=optical_kernel_size,
            optical_sigma=optical_sigma,
        )
