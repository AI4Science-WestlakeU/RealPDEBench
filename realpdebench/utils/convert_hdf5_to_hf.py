#!/usr/bin/env python3
"""
HDF5 to HuggingFace Dataset Conversion Script - V2 (Lazy Slicing, Full Resolution).

This script converts HDF5-based datasets to HuggingFace Datasets format using
the LAZY SLICING approach: store complete trajectories and separate index files.

Key features:
- Stores COMPLETE trajectories at FULL RESOLUTION (no spatial subsampling)
- Each sim_id is stored only ONCE (no overlap)
- Index files (JSON) map (sim_id, time_id) for train/val/test splits
- Supports dynamic N_autoregressive at runtime
- Stores ALL fields: u, v, p, vo, x, y, t (same as H5)

Output structure:
    {dataset_root}/{dataset_name}/hf_dataset/
    ├── real/                          # Arrow: complete trajectories
    │   ├── data-00000-of-XXXXX.arrow
    │   ├── dataset_info.json
    │   └── state.json
    ├── numerical/                     # Arrow: complete trajectories
    │   ├── data-00000-of-XXXXX.arrow
    │   ├── dataset_info.json
    │   └── state.json
    ├── train_index_real.json          # Index files
    ├── val_index_real.json
    ├── test_index_real.json
    ├── train_index_numerical.json
    ├── val_index_numerical.json
    └── test_index_numerical.json

Schema (complete trajectories, full resolution):
    Fluid datasets:
        - sim_id: str
        - u: bytes (np.float32, shape (T_full, H, W))
        - v: bytes (np.float32, shape (T_full, H, W))
        - p: bytes (np.float32, shape (T_full, H, W)) - numerical only
        - vo: bytes (np.float32, shape (T_full, H, W)) - when available
        - x: bytes (np.float64, shape (H, W))  # spatial grid
        - y: bytes (np.float64, shape (H, W))  # spatial grid
        - t: bytes (np.float64, shape (T_full,))  # time array
        - shape_t, shape_h, shape_w: int

    Combustion datasets:
        - sim_id: str
        - observed: bytes (np.float32, shape (T_full, H, W))
        - numerical: bytes (np.float32, shape (T_full, H, W, 15)) - numerical only
        - x: bytes (np.float64, shape (H,))  # 1D grid
        - y: bytes (np.float64, shape (H,))  # 1D grid
        - t: bytes (np.float64, shape (T_full,))  # time array
        - shape_t, shape_h, shape_w: int
        - numerical_channels: int - numerical only

Usage:
    python -m realpdebench.utils.convert_hdf5_to_hf \\
        --dataset_name fsi \\
        --dataset_root /wutailin/real_benchmark/

    # Include all H5 files (even those not in train/val/test splits)
    python -m realpdebench.utils.convert_hdf5_to_hf \\
        --dataset_name fsi \\
        --dataset_root /wutailin/real_benchmark/ \\
        --all_trajectories

    python -m realpdebench.utils.convert_hdf5_to_hf \\
        --dataset_name all \\
        --dataset_root /wutailin/real_benchmark/ \\
        --all_trajectories
"""

import os
import sys
import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Any, Set
from dataclasses import dataclass

import numpy as np
import torch
import h5py
from tqdm import tqdm
from datasets import Dataset, Features, Value


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset type."""
    name: str
    is_combustion: bool
    file_pattern: str
    sub_s_real: int  # Kept for reference in loader, not used during conversion
    sub_s_numerical: int  # Kept for reference in loader, not used during conversion
    real_keys: List[str]
    numerical_keys: List[str]
    surrogate_path: Optional[str] = None
    numerical_channel: int = 0


# Dataset configurations
DATASET_CONFIGS = {
    "combustion": DatasetConfig(
        name="combustion",
        is_combustion=True,
        file_pattern=r"\d+NH3_\d+\.?\d*\.h5",
        sub_s_real=2,
        sub_s_numerical=2,
        real_keys=["trajectory"],
        numerical_keys=["measured_data"],
        surrogate_path="surrogate",
        numerical_channel=15,
    ),
    "cylinder": DatasetConfig(
        name="cylinder",
        is_combustion=False,
        file_pattern=r"\d+\.h5",
        sub_s_real=1,
        sub_s_numerical=2,
        real_keys=["measured_data/u", "measured_data/v"],
        numerical_keys=["measured_data/u", "measured_data/v", "measured_data/p"],
    ),
    "fsi": DatasetConfig(
        name="fsi",
        is_combustion=False,
        file_pattern=r"\d+_[\d\.]+_[\d\.]+_\d+\.h5",
        sub_s_real=2,
        sub_s_numerical=2,
        real_keys=["measured_data/u", "measured_data/v"],
        numerical_keys=["measured_data/u", "measured_data/v", "measured_data/p"],
    ),
    "controlled_cylinder": DatasetConfig(
        name="controlled_cylinder",
        is_combustion=False,
        file_pattern=r"\d+_\d+\.?\d*\.h5",
        sub_s_real=1,
        sub_s_numerical=2,
        real_keys=["measured_data/u", "measured_data/v"],
        numerical_keys=["measured_data/u", "measured_data/v", "measured_data/p"],
    ),
    "foil": DatasetConfig(
        name="foil",
        is_combustion=False,
        file_pattern=r"\d+_\d+\.?\d*\.h5",
        sub_s_real=2,
        sub_s_numerical=2,
        real_keys=["measured_data/u", "measured_data/v"],
        numerical_keys=["measured_data/u", "measured_data/v", "measured_data/p"],
    ),
}


def _to_jsonable(obj: Any) -> Any:
    """
    Convert common Python / NumPy / Torch container types into JSON-serializable objects.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Unsupported type for JSON export: {type(obj)}")


def export_test_params_pt_to_json(
    dataset_dir: str,
    dataset_types: List[str],
    overwrite: bool = False,
) -> Dict[Tuple[str, str], str]:
    """
    Export `*_test_params_*.pt` files to JSON for HF dataset wrappers.
    """
    outputs: Dict[Tuple[str, str], str] = {}
    bases = ["in_dist_test_params", "out_dist_test_params", "remain_params"]

    for dtype in dataset_types:
        for base in bases:
            pt_path = os.path.join(dataset_dir, f"{base}_{dtype}.pt")
            json_path = os.path.join(dataset_dir, f"{base}_{dtype}.json")

            if not os.path.exists(pt_path):
                logging.warning(f"Metadata .pt not found (skip): {pt_path}")
                continue
            if os.path.exists(json_path) and not overwrite:
                logging.info(f"Metadata JSON already exists (skip): {json_path}")
                outputs[(base, dtype)] = json_path
                continue

            data = torch.load(pt_path, weights_only=False)
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict in {pt_path}, got {type(data)}")

            sorted_items = {k: data[k] for k in sorted(data.keys())}
            payload = _to_jsonable(sorted_items)

            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)

            outputs[(base, dtype)] = json_path
            logging.info(f"Wrote metadata JSON: {json_path}")

    return outputs


def load_mapping_files(dataset_dir: str, dataset_type: str) -> Tuple[Dict, Dict]:
    """Load pre-computed sim_id and time_id mappings from .pt files."""
    sim_id_path = os.path.join(dataset_dir, f"sim_id_mapping_{dataset_type}.pt")
    time_id_path = os.path.join(dataset_dir, f"time_id_mapping_{dataset_type}.pt")

    if not os.path.exists(sim_id_path):
        raise FileNotFoundError(f"Mapping file not found: {sim_id_path}")
    if not os.path.exists(time_id_path):
        raise FileNotFoundError(f"Mapping file not found: {time_id_path}")

    sim_id_mapping = torch.load(sim_id_path, weights_only=False)
    time_id_mapping = torch.load(time_id_path, weights_only=False)

    return sim_id_mapping, time_id_mapping


def get_unique_sim_ids(sim_id_mapping: Dict[str, List[str]]) -> Set[str]:
    """Get all unique sim_ids across all splits."""
    unique_ids = set()
    for split in ["train", "val", "test"]:
        if split in sim_id_mapping:
            unique_ids.update(sim_id_mapping[split])
    return unique_ids


def get_all_h5_sim_ids(data_path: str) -> List[str]:
    """Get all H5 files from directory (for --all_trajectories mode)."""
    if not os.path.isdir(data_path):
        return []
    return [f for f in os.listdir(data_path) if f.endswith('.h5')]


def fluid_trajectory_generator(
    data_path: str,
    sim_ids: List[str],
    is_numerical: bool,
    dataset_name: str,
) -> Iterator[Dict[str, Any]]:
    """
    Generator for complete fluid trajectories at FULL RESOLUTION.

    Stores: u, v, p (numerical), vo (when available), x, y, t
    """
    for sim_id in tqdm(sim_ids, desc="Loading trajectories"):
        h5_path = os.path.join(data_path, sim_id)

        try:
            with h5py.File(h5_path, "r") as f:
                # Load velocity fields at FULL RESOLUTION (no subsampling)
                u = f["measured_data"]["u"][:]  # (T, H, W)
                v = f["measured_data"]["v"][:]  # (T, H, W)

                # Pressure field (numerical only)
                p = None
                if is_numerical and "p" in f["measured_data"]:
                    p = f["measured_data"]["p"][:]  # (T, H, W)

                # Vorticity field (when available)
                vo = None
                if "vo" in f:
                    vo = f["vo"][:]  # (T, H, W)
                elif "vo" in f.get("measured_data", {}):
                    vo = f["measured_data"]["vo"][:]  # (T, H, W)

                # Time array
                if "t" in f:
                    t = f["t"][:]  # (T,)
                elif "t" in f.get("measured_data", {}):
                    t = f["measured_data"]["t"][:]  # (T,)
                else:
                    t = None

                # Spatial grids x, y
                # Some datasets have (T, H, W) shape, we take [0] for static grid
                # Some have (H, W) shape directly
                x = f["x"][:]
                y = f["y"][:]

                if x.ndim == 3:  # (T, H, W) -> take first frame
                    x = x[0]  # (H, W)
                if y.ndim == 3:  # (T, H, W) -> take first frame
                    y = y[0]  # (H, W)

            # Convert to appropriate dtypes
            u = u.astype(np.float32)
            v = v.astype(np.float32)

            record = {
                "sim_id": sim_id,
                "u": u.tobytes(),
                "v": v.tobytes(),
                "shape_t": u.shape[0],
                "shape_h": u.shape[1],
                "shape_w": u.shape[2],
            }

            if p is not None:
                record["p"] = p.astype(np.float32).tobytes()

            if vo is not None:
                record["vo"] = vo.astype(np.float32).tobytes()

            if x is not None:
                record["x"] = x.astype(np.float64).tobytes()
                record["x_shape_h"] = x.shape[0]
                record["x_shape_w"] = x.shape[1] if x.ndim > 1 else 0

            if y is not None:
                record["y"] = y.astype(np.float64).tobytes()
                record["y_shape_h"] = y.shape[0]
                record["y_shape_w"] = y.shape[1] if y.ndim > 1 else 0

            if t is not None:
                record["t"] = t.astype(np.float64).tobytes()
                record["t_shape"] = t.shape[0]

            yield record

        except Exception as e:
            logging.warning(f"Error loading {h5_path}: {e}")
            continue


def combustion_trajectory_generator(
    data_path: str,
    surrogate_path: Optional[str],
    sim_ids: List[str],
    is_numerical: bool,
    numerical_channel: int,
) -> Iterator[Dict[str, Any]]:
    """
    Generator for complete combustion trajectories at FULL RESOLUTION.

    Stores: observed, numerical (when numerical), x, y, t
    """
    for sim_id in tqdm(sim_ids, desc="Loading trajectories"):
        try:
            if is_numerical:
                # Load from surrogate and numerical paths
                surrogate_h5 = os.path.join(surrogate_path, sim_id)
                numerical_h5 = os.path.join(data_path, sim_id)

                with h5py.File(surrogate_h5, "r") as f:
                    observed = f["measured_data"][:]  # (T, H, W) - full resolution
                with h5py.File(numerical_h5, "r") as f:
                    numerical = f["measured_data"][:]  # (T, H, W, 15)
                    # Get spatial/temporal grids from numerical file
                    x = f["x"][:] if "x" in f else None  # (H,) 1D
                    y = f["y"][:] if "y" in f else None  # (H,) 1D
                    t = f["time"][:] if "time" in f else None  # (T,)
            else:
                # Load from real path
                real_h5 = os.path.join(data_path, sim_id)
                with h5py.File(real_h5, "r") as f:
                    observed = f["trajectory"][:]  # (T, H, W) - full resolution
                    # Real combustion H5 may not have x, y, t
                    x = f["x"][:] if "x" in f else None
                    y = f["y"][:] if "y" in f else None
                    t = f["time"][:] if "time" in f else None
                numerical = None

            observed = observed.astype(np.float32)

            record = {
                "sim_id": sim_id,
                "observed": observed.tobytes(),
                "shape_t": observed.shape[0],
                "shape_h": observed.shape[1],
                "shape_w": observed.shape[2],
            }

            if numerical is not None:
                numerical = numerical.astype(np.float32)
                record["numerical"] = numerical.tobytes()
                record["numerical_channels"] = numerical.shape[-1]

            if x is not None:
                record["x"] = x.astype(np.float64).tobytes()
                record["x_shape"] = x.shape[0]

            if y is not None:
                record["y"] = y.astype(np.float64).tobytes()
                record["y_shape"] = y.shape[0]

            if t is not None:
                record["t"] = t.astype(np.float64).tobytes()
                record["t_shape"] = t.shape[0]

            yield record

        except Exception as e:
            logging.warning(f"Error loading {sim_id}: {e}")
            continue


def combustion_surrogate_train_generator(
    real_surrogate_path: str,
    numerical_surrogate_path: str,
    sim_ids: List[str],
    time_ids: List[int],
    step: int,
    sub_s_real: int,
    sub_s_numerical: int,
) -> Iterator[Dict[str, Any]]:
    """
    Generator for combustion surrogate-train dataset samples.

    NOTE: This still uses subsampling for backward compatibility with
    the surrogate training pipeline.
    """
    for sim_id in tqdm(sim_ids, desc="Generating surrogate_train"):
        real_h5_path = os.path.join(real_surrogate_path, sim_id)
        numerical_h5_path = os.path.join(numerical_surrogate_path, sim_id)
        try:
            with h5py.File(real_h5_path, "r") as f_real, h5py.File(numerical_h5_path, "r") as f_num:
                for time_id in time_ids:
                    real = f_real["trajectory"][
                        time_id:time_id + step, ::sub_s_real, ::sub_s_real
                    ]
                    numerical = f_num["measured_data"][
                        time_id:time_id + step, ::sub_s_numerical, ::sub_s_numerical
                    ]

                    real = real.astype(np.float32)
                    numerical = numerical.astype(np.float32)

                    if numerical.ndim != 4:
                        raise ValueError(
                            f"Expected numerical surrogate data to have 4 dims (T,H,W,C), "
                            f"but got shape={numerical.shape} for {numerical_h5_path}"
                        )

                    record = {
                        "sim_id": sim_id,
                        "time_id": int(time_id),
                        "real": real.tobytes(),
                        "numerical": numerical.tobytes(),
                        "real_shape_t": int(real.shape[0]),
                        "real_shape_h": int(real.shape[1]),
                        "real_shape_w": int(real.shape[2]),
                        "numerical_shape_t": int(numerical.shape[0]),
                        "numerical_shape_h": int(numerical.shape[1]),
                        "numerical_shape_w": int(numerical.shape[2]),
                        "numerical_channels": int(numerical.shape[3]),
                    }
                    yield record
        except Exception as e:
            logging.warning(f"Error loading surrogate_train {sim_id}: {e}")
            continue


def get_fluid_features(is_numerical: bool) -> Features:
    """Get HF Dataset features for fluid trajectories (full resolution)."""
    features = {
        "sim_id": Value("string"),
        "u": Value("binary"),
        "v": Value("binary"),
        "shape_t": Value("int32"),
        "shape_h": Value("int32"),
        "shape_w": Value("int32"),
        # Optional fields (may be None for some records)
        "vo": Value("binary"),
        "x": Value("binary"),
        "x_shape_h": Value("int32"),
        "x_shape_w": Value("int32"),
        "y": Value("binary"),
        "y_shape_h": Value("int32"),
        "y_shape_w": Value("int32"),
        "t": Value("binary"),
        "t_shape": Value("int32"),
    }
    if is_numerical:
        features["p"] = Value("binary")
    return Features(features)


def get_combustion_features(is_numerical: bool) -> Features:
    """Get HF Dataset features for combustion trajectories (full resolution)."""
    features = {
        "sim_id": Value("string"),
        "observed": Value("binary"),
        "shape_t": Value("int32"),
        "shape_h": Value("int32"),
        "shape_w": Value("int32"),
        # Optional fields
        "x": Value("binary"),
        "x_shape": Value("int32"),
        "y": Value("binary"),
        "y_shape": Value("int32"),
        "t": Value("binary"),
        "t_shape": Value("int32"),
    }
    if is_numerical:
        features["numerical"] = Value("binary")
        features["numerical_channels"] = Value("int32")
    return Features(features)


def get_combustion_surrogate_train_features() -> Features:
    """Get HF Dataset features for combustion surrogate-train dataset."""
    return Features(
        {
            "sim_id": Value("string"),
            "time_id": Value("int32"),
            "real": Value("binary"),
            "numerical": Value("binary"),
            "real_shape_t": Value("int32"),
            "real_shape_h": Value("int32"),
            "real_shape_w": Value("int32"),
            "numerical_shape_t": Value("int32"),
            "numerical_shape_h": Value("int32"),
            "numerical_shape_w": Value("int32"),
            "numerical_channels": Value("int32"),
        }
    )


def generate_index_files(
    dataset_dir: str,
    dataset_type: str,
    output_dir: str,
    splits: List[str],
) -> Dict[str, str]:
    """Generate JSON index files from existing mapping files."""
    sim_id_mapping, time_id_mapping = load_mapping_files(dataset_dir, dataset_type)

    output_files = {}

    for split in splits:
        sim_ids = sim_id_mapping.get(split, [])
        time_ids = time_id_mapping.get(split, [])

        if len(sim_ids) == 0:
            logging.info(f"  Skipping index for {split}: no samples")
            continue

        # Create index list
        indices = [
            {"sim_id": sim_id, "time_id": int(time_id)}
            for sim_id, time_id in zip(sim_ids, time_ids)
        ]

        # Write to JSON
        output_path = os.path.join(output_dir, f"{split}_index_{dataset_type}.json")
        with open(output_path, "w") as f:
            json.dump(indices, f)

        output_files[split] = output_path
        logging.info(f"  Index {split}: {len(indices)} entries -> {output_path}")

    return output_files


def convert_fluid_to_hf_v2(
    dataset_name: str,
    dataset_root: str,
    output_dir: str,
    config: DatasetConfig,
    dataset_types: List[str],
    splits: List[str],
    max_shard_size: str,
    all_trajectories: bool,
) -> Dict[str, str]:
    """Convert fluid dataset to HF format with lazy slicing and full resolution."""
    dataset_dir = os.path.join(dataset_root, dataset_name)
    output_files = {}

    for dtype in dataset_types:
        data_path = os.path.join(dataset_dir, dtype)
        is_numerical = dtype == "numerical"

        # Determine which sim_ids to convert
        if all_trajectories:
            # Get all H5 files from directory
            all_sim_ids = get_all_h5_sim_ids(data_path)
            if len(all_sim_ids) == 0:
                logging.warning(f"Skipping {dataset_name}/{dtype}: no H5 files in {data_path}")
                continue
            unique_sim_ids = sorted(all_sim_ids)
            logging.info(f"  Using --all_trajectories: found {len(unique_sim_ids)} H5 files")
        else:
            # Load mappings to get unique sim_ids
            try:
                sim_id_mapping, _ = load_mapping_files(dataset_dir, dtype)
            except FileNotFoundError as e:
                logging.warning(f"Skipping {dataset_name}/{dtype}: {e}")
                continue
            unique_sim_ids = sorted(get_unique_sim_ids(sim_id_mapping))

        if len(unique_sim_ids) == 0:
            logging.info(f"Skipping {dataset_name}/{dtype}: no trajectories")
            continue

        logging.info(f"Converting {dataset_name}/{dtype}: {len(unique_sim_ids)} trajectories (full resolution)")

        # Create generator function with closure
        def make_generator(
            data_path=data_path,
            sim_ids=unique_sim_ids,
            is_numerical=is_numerical,
            dataset_name=dataset_name,
        ):
            return fluid_trajectory_generator(
                data_path=data_path,
                sim_ids=sim_ids,
                is_numerical=is_numerical,
                dataset_name=dataset_name,
            )

        # Create HF Dataset from generator
        features = get_fluid_features(is_numerical)
        dataset = Dataset.from_generator(
            make_generator,
            features=features,
        )

        # Save trajectory data
        traj_output_path = os.path.join(output_dir, dtype)
        dataset.save_to_disk(traj_output_path, max_shard_size=max_shard_size)
        output_files[f"{dtype}_trajectories"] = traj_output_path
        logging.info(f"  Saved {len(dataset)} trajectories to {traj_output_path}")

        # Generate index files (always use mapping files, not all_trajectories)
        try:
            index_files = generate_index_files(
                dataset_dir=dataset_dir,
                dataset_type=dtype,
                output_dir=output_dir,
                splits=splits,
            )
            output_files.update({f"{dtype}_{k}_index": v for k, v in index_files.items()})
        except FileNotFoundError as e:
            logging.warning(f"Could not generate index files for {dataset_name}/{dtype}: {e}")

    return output_files


def convert_combustion_to_hf_v2(
    dataset_name: str,
    dataset_root: str,
    output_dir: str,
    config: DatasetConfig,
    dataset_types: List[str],
    splits: List[str],
    max_shard_size: str,
    all_trajectories: bool,
) -> Dict[str, str]:
    """Convert combustion dataset to HF format with lazy slicing and full resolution."""
    dataset_dir = os.path.join(dataset_root, dataset_name)
    surrogate_path = os.path.join(dataset_dir, config.surrogate_path) if config.surrogate_path else None
    output_files = {}

    for dtype in dataset_types:
        data_path = os.path.join(dataset_dir, dtype)
        is_numerical = dtype == "numerical"

        # Determine which sim_ids to convert
        if all_trajectories:
            all_sim_ids = get_all_h5_sim_ids(data_path)
            if len(all_sim_ids) == 0:
                logging.warning(f"Skipping {dataset_name}/{dtype}: no H5 files in {data_path}")
                continue
            unique_sim_ids = sorted(all_sim_ids)
            logging.info(f"  Using --all_trajectories: found {len(unique_sim_ids)} H5 files")
        else:
            try:
                sim_id_mapping, _ = load_mapping_files(dataset_dir, dtype)
            except FileNotFoundError as e:
                logging.warning(f"Skipping {dataset_name}/{dtype}: {e}")
                continue
            unique_sim_ids = sorted(get_unique_sim_ids(sim_id_mapping))

        if len(unique_sim_ids) == 0:
            logging.info(f"Skipping {dataset_name}/{dtype}: no trajectories")
            continue

        logging.info(f"Converting {dataset_name}/{dtype}: {len(unique_sim_ids)} trajectories (full resolution)")

        # Create generator function with closure
        def make_generator(
            data_path=data_path,
            surrogate_path=surrogate_path,
            sim_ids=unique_sim_ids,
            is_numerical=is_numerical,
            numerical_channel=config.numerical_channel,
        ):
            return combustion_trajectory_generator(
                data_path=data_path,
                surrogate_path=surrogate_path,
                sim_ids=sim_ids,
                is_numerical=is_numerical,
                numerical_channel=numerical_channel,
            )

        # Create HF Dataset from generator
        features = get_combustion_features(is_numerical)
        dataset = Dataset.from_generator(
            make_generator,
            features=features,
        )

        # Save trajectory data
        traj_output_path = os.path.join(output_dir, dtype)
        dataset.save_to_disk(traj_output_path, max_shard_size=max_shard_size)
        output_files[f"{dtype}_trajectories"] = traj_output_path
        logging.info(f"  Saved {len(dataset)} trajectories to {traj_output_path}")

        # Generate index files
        try:
            index_files = generate_index_files(
                dataset_dir=dataset_dir,
                dataset_type=dtype,
                output_dir=output_dir,
                splits=splits,
            )
            output_files.update({f"{dtype}_{k}_index": v for k, v in index_files.items()})
        except FileNotFoundError as e:
            logging.warning(f"Could not generate index files for {dataset_name}/{dtype}: {e}")

    return output_files


def convert_combustion_surrogate_train_to_hf(
    dataset_root: str,
    output_dir: Optional[str],
    max_shard_size: str,
    num_proc: Optional[int],
    step: int,
    n_sim_frame: int,
    sub_s_real: int,
    sub_s_numerical: int,
) -> Optional[str]:
    """
    Convert combustion surrogate-train (HDF5) to HF Arrow-sharded dataset.

    NOTE: This still uses subsampling for backward compatibility.
    """
    dataset_name = "combustion"
    dataset_dir = os.path.join(dataset_root, dataset_name)
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, "hf_dataset")
    os.makedirs(output_dir, exist_ok=True)

    real_surrogate_path = os.path.join(dataset_dir, "real_surrogate_train")
    numerical_surrogate_path = os.path.join(dataset_dir, "numerical_surrogate_train")

    if not os.path.isdir(real_surrogate_path) or not os.path.isdir(numerical_surrogate_path):
        logging.warning(
            "Skipping combustion surrogate_train conversion: expected folders not found:\n"
            f"  - {real_surrogate_path}\n"
            f"  - {numerical_surrogate_path}"
        )
        return None

    sim_ids = [f for f in os.listdir(numerical_surrogate_path) if f.endswith(".h5")]
    if len(sim_ids) == 0:
        logging.warning(
            f"Skipping combustion surrogate_train conversion: no .h5 files in {numerical_surrogate_path}"
        )
        return None

    if n_sim_frame <= step:
        raise ValueError(
            f"Invalid surrogate_train windowing: n_sim_frame={n_sim_frame} must be > step={step}"
        )

    time_ids = list(range(n_sim_frame - step))

    logging.info(
        f"Converting combustion surrogate_train: {len(sim_ids)} sims × {len(time_ids)} time windows "
        f"(step={step}, n_sim_frame={n_sim_frame})"
    )

    # Save sim order + meta
    sim_ids_path = os.path.join(output_dir, "surrogate_train_sim_ids.txt")
    with open(sim_ids_path, "w") as f:
        for sim_id in sim_ids:
            f.write(f"{sim_id}\n")

    meta_path = os.path.join(output_dir, "surrogate_train_meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "step": int(step),
                "n_sim_frame": int(n_sim_frame),
                "sub_s_real": int(sub_s_real),
                "sub_s_numerical": int(sub_s_numerical),
                "sim_ids_file": os.path.basename(sim_ids_path),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    output_path = os.path.join(output_dir, "surrogate_train")
    if os.path.exists(output_path):
        logging.info(f"Skipping combustion surrogate_train: output already exists at {output_path}")
        return output_path

    def make_generator(
        real_surrogate_path=real_surrogate_path,
        numerical_surrogate_path=numerical_surrogate_path,
        sim_ids=sim_ids,
        time_ids=time_ids,
        step=step,
        sub_s_real=sub_s_real,
        sub_s_numerical=sub_s_numerical,
    ):
        return combustion_surrogate_train_generator(
            real_surrogate_path=real_surrogate_path,
            numerical_surrogate_path=numerical_surrogate_path,
            sim_ids=sim_ids,
            time_ids=time_ids,
            step=step,
            sub_s_real=sub_s_real,
            sub_s_numerical=sub_s_numerical,
        )

    features = get_combustion_surrogate_train_features()
    dataset = Dataset.from_generator(
        make_generator,
        features=features,
        num_proc=num_proc,
    )
    dataset.save_to_disk(output_path, max_shard_size=max_shard_size)
    logging.info(f"  Saved {len(dataset)} surrogate_train samples to {output_path}")

    return output_path


def convert_dataset_v2(
    dataset_name: str,
    dataset_root: str,
    output_dir: Optional[str],
    config: DatasetConfig,
    dataset_types: List[str],
    splits: List[str],
    max_shard_size: str,
    all_trajectories: bool,
) -> Dict[str, str]:
    """Convert a dataset from HDF5 to HF Dataset format (V2 lazy slicing, full resolution)."""
    if output_dir is None:
        output_dir = os.path.join(dataset_root, dataset_name, "hf_dataset")

    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Converting dataset: {dataset_name} (V2 lazy slicing, full resolution)")
    logging.info(f"  Output: {output_dir}")
    logging.info(f"  Max shard size: {max_shard_size}")
    logging.info(f"  All trajectories: {all_trajectories}")

    if config.is_combustion:
        return convert_combustion_to_hf_v2(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            output_dir=output_dir,
            config=config,
            dataset_types=dataset_types,
            splits=splits,
            max_shard_size=max_shard_size,
            all_trajectories=all_trajectories,
        )
    else:
        return convert_fluid_to_hf_v2(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            output_dir=output_dir,
            config=config,
            dataset_types=dataset_types,
            splits=splits,
            max_shard_size=max_shard_size,
            all_trajectories=all_trajectories,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 datasets to HuggingFace Dataset format (V2 lazy slicing, full resolution)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()) + ["all"],
        help="Name of the dataset to convert, or 'all' for all datasets",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/wutailin/real_benchmark/",
        help="Root directory containing datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: {dataset_root}/{dataset_name}/hf_dataset)",
    )
    parser.add_argument(
        "--dataset_types",
        type=str,
        nargs="+",
        default=["real", "numerical"],
        help="Dataset types to convert",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Data splits to generate indices for",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="500MB",
        help="Maximum shard size (e.g., '500MB', '1GB')",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--all_trajectories",
        action="store_true",
        help=(
            "Include ALL H5 files from directory, not just those in mapping files. "
            "Useful for datasets where mapping only covers a subset (e.g., fsi 45/51 issue). "
            "Index files will still use mapping-based splits."
        ),
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes for surrogate-train conversion",
    )
    parser.add_argument(
        "--export_test_params_json",
        action="store_true",
        help=(
            "Export test_mode metadata from `*_test_params_*.pt` to JSON "
            "(writes `*_test_params_*.json` next to the .pt files)."
        ),
    )
    parser.add_argument(
        "--overwrite_test_params_json",
        action="store_true",
        help="Overwrite existing `*_test_params_*.json` files when exporting.",
    )
    parser.add_argument(
        "--include_surrogate_train",
        action="store_true",
        help=(
            "Also convert combustion surrogate-train dataset "
            "(real_surrogate_train + numerical_surrogate_train) into HF Arrow shards."
        ),
    )
    parser.add_argument(
        "--surrogate_step",
        type=int,
        default=20,
        help="Window length for surrogate-train conversion (matches SurrogateDataset.step)",
    )
    parser.add_argument(
        "--surrogate_n_sim_frame",
        type=int,
        default=40,
        help="n_sim_frame for surrogate-train conversion (matches SurrogateDataset.n_sim_frame)",
    )
    parser.add_argument(
        "--surrogate_sub_s_real",
        type=int,
        default=1,
        help="Spatial subsampling for real_surrogate_train during conversion",
    )
    parser.add_argument(
        "--surrogate_sub_s_numerical",
        type=int,
        default=1,
        help="Spatial subsampling for numerical_surrogate_train during conversion",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Determine which datasets to convert
    if args.dataset_name == "all":
        dataset_names = list(DATASET_CONFIGS.keys())
    else:
        dataset_names = [args.dataset_name]

    logging.info(f"Dataset root: {args.dataset_root}")
    logging.info(f"Datasets to convert: {dataset_names}")
    logging.info(f"Mode: V2 (lazy slicing, full resolution)")

    all_output_files = {}

    for name in dataset_names:
        config = DATASET_CONFIGS[name]

        # Determine output directory
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = os.path.join(args.dataset_root, name, "hf_dataset")

        try:
            output_files = convert_dataset_v2(
                dataset_name=name,
                dataset_root=args.dataset_root,
                output_dir=output_dir,
                config=config,
                dataset_types=args.dataset_types,
                splits=args.splits,
                max_shard_size=args.max_shard_size,
                all_trajectories=args.all_trajectories,
            )
            all_output_files.update({(name, k): v for k, v in output_files.items()})

            if args.export_test_params_json:
                try:
                    dataset_dir = os.path.join(args.dataset_root, name)
                    export_test_params_pt_to_json(
                        dataset_dir=dataset_dir,
                        dataset_types=args.dataset_types,
                        overwrite=args.overwrite_test_params_json,
                    )
                except Exception as e:
                    logging.error(f"Error exporting test params JSON for {name}: {e}")
                    continue

            if args.include_surrogate_train and name == "combustion":
                try:
                    surrogate_out = convert_combustion_surrogate_train_to_hf(
                        dataset_root=args.dataset_root,
                        output_dir=output_dir,
                        max_shard_size=args.max_shard_size,
                        num_proc=args.num_proc,
                        step=args.surrogate_step,
                        n_sim_frame=args.surrogate_n_sim_frame,
                        sub_s_real=args.surrogate_sub_s_real,
                        sub_s_numerical=args.surrogate_sub_s_numerical,
                    )
                    if surrogate_out is not None:
                        all_output_files[(name, "surrogate_train")] = surrogate_out
                except Exception as e:
                    logging.error(f"Error converting {name}/surrogate_train: {e}")
                    continue
        except Exception as e:
            logging.error(f"Error converting {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logging.info("=" * 60)
    logging.info("Conversion complete!")
    logging.info("Output files:")
    for key, path in all_output_files.items():
        logging.info(f"  {key}: {path}")


if __name__ == "__main__":
    main()
