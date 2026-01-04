#!/usr/bin/env python3
"""
HDF5 to HuggingFace Dataset Conversion Script for RealBench Datasets.

This script converts HDF5-based datasets to HuggingFace Datasets format with
automatic sharding (~500MB per shard) for Hub compatibility.

Unlike convert_hdf5_to_parquet.py which produces raw Parquet files, this script
uses HF Datasets' native functions for proper sharding and metadata.

Usage:
    # Convert a single dataset
    python scripts/convert_hdf5_to_hf.py \
        --dataset_name fluid_single_cylinder \
        --dataset_root /wutailin/real_benchmark/

    # Convert all datasets
    python scripts/convert_hdf5_to_hf.py \
        --dataset_name all \
        --dataset_root /wutailin/real_benchmark/

    # Convert with custom shard size (in bytes)
    python scripts/convert_hdf5_to_hf.py \
        --dataset_name fluid_single_cylinder \
        --dataset_root /wutailin/real_benchmark/ \
        --max_shard_size 500MB

Supported datasets:
    - combustion (single_injector_128)
    - cylinder (fluid_single_cylinder)
    - fsi (fluid_double_cylinder)
    - controlled_cylinder (fluid_control)
    - foil (fluid_foil)

Output structure:
    {dataset_root}/{dataset_name}/hf_dataset/
    ├── real_train/
    │   ├── data-00000-of-00005.arrow
    │   ├── data-00001-of-00005.arrow
    │   ├── ...
    │   ├── dataset_info.json
    │   └── state.json
    ├── real_val/
    ├── real_test/
    ├── numerical_train/
    ├── numerical_val/
    └── numerical_test/

Schema (same as Parquet approach):
    Fluid datasets:
        - sim_id: str
        - time_id: int
        - u: bytes (np.float32, shape (horizon, H, W))
        - v: bytes (np.float32, shape (horizon, H, W))
        - p: bytes (np.float32, shape (horizon, H, W)) - numerical only
        - shape_t, shape_h, shape_w: int

    Combustion datasets:
        - sim_id: str
        - time_id: int
        - observed: bytes (np.float32, shape (horizon, H, W))
        - numerical: bytes (np.float32, shape (horizon, H, W, 15)) - numerical only
        - shape_t, shape_h, shape_w: int
        - numerical_channels: int - numerical only
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Any
from dataclasses import dataclass

import numpy as np
import torch
import h5py
from tqdm import tqdm
from datasets import Dataset, Features, Value, Sequence


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset type."""
    name: str
    is_combustion: bool
    file_pattern: str
    sub_s_real: int
    sub_s_numerical: int
    horizon: int
    real_keys: List[str]
    numerical_keys: List[str]
    surrogate_path: Optional[str] = None
    numerical_channel: int = 0


# Dataset configurations (same as convert_hdf5_to_parquet.py)
DATASET_CONFIGS = {
    "combustion": DatasetConfig(
        name="combustion",
        is_combustion=True,
        file_pattern=r"\d+NH3_\d+\.?\d*\.h5",
        sub_s_real=2,
        sub_s_numerical=2,
        horizon=40,
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
        horizon=40,
        real_keys=["measured_data/u", "measured_data/v"],
        numerical_keys=["measured_data/u", "measured_data/v", "measured_data/p"],
    ),
    "fsi": DatasetConfig(
        name="fsi",
        is_combustion=False,
        file_pattern=r"\d+_[\d\.]+_[\d\.]+_\d+\.h5",
        sub_s_real=2,
        sub_s_numerical=2,
        horizon=40,
        real_keys=["measured_data/u", "measured_data/v"],
        numerical_keys=["measured_data/u", "measured_data/v", "measured_data/p"],
    ),
    "controlled_cylinder": DatasetConfig(
        name="controlled_cylinder",
        is_combustion=False,
        file_pattern=r"\d+_\d+\.?\d*\.h5",
        sub_s_real=1,
        sub_s_numerical=2,
        horizon=20,
        real_keys=["measured_data/u", "measured_data/v"],
        numerical_keys=["measured_data/u", "measured_data/v", "measured_data/p"],
    ),
    "foil": DatasetConfig(
        name="foil",
        is_combustion=False,
        file_pattern=r"\d+_\d+\.?\d*\.h5",
        sub_s_real=2,
        sub_s_numerical=2,
        horizon=40,
        real_keys=["measured_data/u", "measured_data/v"],
        numerical_keys=["measured_data/u", "measured_data/v", "measured_data/p"],
    ),
}


def _to_jsonable(obj: Any) -> Any:
    """
    Convert common Python / NumPy / Torch container types into JSON-serializable objects.
    
    This is intentionally strict: raise on unknown types to avoid silently writing
    incorrect metadata.
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
    Export `*_test_params_*.pt` files (created by original HDF5 datasets) to JSON.
    
    HF dataset wrappers use these to implement `test_mode` filtering
    (`in_dist`, `out_dist`, `seen`, `unseen`) for val/test.
    
    Output files (written next to the `.pt` files):
    - `in_dist_test_params_{dataset_type}.json`
    - `out_dist_test_params_{dataset_type}.json`
    - `remain_params_{dataset_type}.json`
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
            
            # Sort keys for deterministic JSON output.
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


def load_fluid_sample(
    h5_path: str,
    time_id: int,
    horizon: int,
    sub_s: int,
    is_numerical: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load a fluid dataset sample from HDF5."""
    with h5py.File(h5_path, "r") as f:
        u = f["measured_data"]["u"][time_id:time_id + horizon, ::sub_s, ::sub_s]
        v = f["measured_data"]["v"][time_id:time_id + horizon, ::sub_s, ::sub_s]
        
        if is_numerical:
            p = f["measured_data"]["p"][time_id:time_id + horizon, ::sub_s, ::sub_s]
        else:
            p = None
    
    return u, v, p


def load_combustion_sample(
    real_path: Optional[str],
    surrogate_path: Optional[str],
    numerical_path: Optional[str],
    time_id: int,
    horizon: int,
    sub_s: int,
    is_numerical: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a combustion dataset sample from HDF5."""
    if is_numerical:
        with h5py.File(surrogate_path, "r") as f:
            observed = f["measured_data"][time_id:time_id + horizon, ::sub_s, ::sub_s]
        with h5py.File(numerical_path, "r") as f:
            numerical = f["measured_data"][time_id:time_id + horizon, ::sub_s, ::sub_s]
    else:
        with h5py.File(real_path, "r") as f:
            observed = f["trajectory"][time_id:time_id + horizon, ::sub_s, ::sub_s]
        numerical = None
    
    return observed, numerical


def fluid_generator(
    dataset_dir: str,
    data_path: str,
    sim_ids: List[str],
    time_ids: List[int],
    horizon: int,
    sub_s: int,
    is_numerical: bool,
) -> Iterator[Dict[str, Any]]:
    """Generator for fluid dataset samples."""
    for sim_id, time_id in tqdm(
        zip(sim_ids, time_ids),
        total=len(sim_ids),
        desc="Generating",
    ):
        h5_path = os.path.join(data_path, sim_id)
        
        try:
            u, v, p = load_fluid_sample(
                h5_path=h5_path,
                time_id=time_id,
                horizon=horizon,
                sub_s=sub_s,
                is_numerical=is_numerical,
            )
            
            u = u.astype(np.float32)
            v = v.astype(np.float32)
            
            record = {
                "sim_id": sim_id,
                "time_id": time_id,
                "u": u.tobytes(),
                "v": v.tobytes(),
                "shape_t": u.shape[0],
                "shape_h": u.shape[1],
                "shape_w": u.shape[2],
            }
            
            if p is not None:
                record["p"] = p.astype(np.float32).tobytes()
            
            yield record
            
        except Exception as e:
            logging.warning(f"Error loading {h5_path} at time_id={time_id}: {e}")
            continue


def combustion_generator(
    dataset_dir: str,
    data_path: str,
    surrogate_path: Optional[str],
    sim_ids: List[str],
    time_ids: List[int],
    horizon: int,
    sub_s: int,
    is_numerical: bool,
) -> Iterator[Dict[str, Any]]:
    """Generator for combustion dataset samples."""
    for sim_id, time_id in tqdm(
        zip(sim_ids, time_ids),
        total=len(sim_ids),
        desc="Generating",
    ):
        if is_numerical:
            real_path = None
            surrogate_h5 = os.path.join(surrogate_path, sim_id)
            numerical_h5 = os.path.join(data_path, sim_id)
        else:
            real_path = os.path.join(data_path, sim_id)
            surrogate_h5 = None
            numerical_h5 = None
        
        try:
            observed, numerical = load_combustion_sample(
                real_path=real_path,
                surrogate_path=surrogate_h5,
                numerical_path=numerical_h5,
                time_id=time_id,
                horizon=horizon,
                sub_s=sub_s,
                is_numerical=is_numerical,
            )
            
            observed = observed.astype(np.float32)
            
            record = {
                "sim_id": sim_id,
                "time_id": time_id,
                "observed": observed.tobytes(),
                "shape_t": observed.shape[0],
                "shape_h": observed.shape[1],
                "shape_w": observed.shape[2],
            }
            
            if numerical is not None:
                numerical = numerical.astype(np.float32)
                record["numerical"] = numerical.tobytes()
                record["numerical_channels"] = numerical.shape[-1]
            
            yield record
            
        except Exception as e:
            logging.warning(f"Error loading {sim_id} at time_id={time_id}: {e}")
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
    
    Mirrors `data/combustion_surrogate_dataset.py` behavior:
    - real output comes from `real_surrogate_train/{sim_id}.h5` key `trajectory`
    - numerical input comes from `numerical_surrogate_train/{sim_id}.h5` key `measured_data`
    - each sample is a window of length `step` starting at `time_id`
    - spatial subsampling is applied here (like other HF conversions)
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
    """Get HF Dataset features for fluid datasets."""
    features = {
        "sim_id": Value("string"),
        "time_id": Value("int32"),
        "u": Value("binary"),
        "v": Value("binary"),
        "shape_t": Value("int32"),
        "shape_h": Value("int32"),
        "shape_w": Value("int32"),
    }
    if is_numerical:
        features["p"] = Value("binary")
    return Features(features)


def get_combustion_features(is_numerical: bool) -> Features:
    """Get HF Dataset features for combustion datasets."""
    features = {
        "sim_id": Value("string"),
        "time_id": Value("int32"),
        "observed": Value("binary"),
        "shape_t": Value("int32"),
        "shape_h": Value("int32"),
        "shape_w": Value("int32"),
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
    
    Input folders (under `{dataset_root}/combustion/`):
    - `real_surrogate_train/`
    - `numerical_surrogate_train/`
    
    Output folder:
    - `{dataset_root}/combustion/hf_dataset/surrogate_train/`
    
    Also writes:
    - `{dataset_root}/combustion/hf_dataset/surrogate_train_sim_ids.txt`
    - `{dataset_root}/combustion/hf_dataset/surrogate_train_meta.json`
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
    
    time_ids = list(range(n_sim_frame - step))  # mirrors combustion_surrogate_dataset.py
    
    logging.info(
        f"Converting combustion surrogate_train: {len(sim_ids)} sims × {len(time_ids)} time windows "
        f"(step={step}, n_sim_frame={n_sim_frame})"
    )
    
    # Save sim order + meta for deterministic HF indexing without depending on HDF5 folder listing.
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


def convert_fluid_to_hf(
    dataset_name: str,
    dataset_root: str,
    output_dir: str,
    config: DatasetConfig,
    dataset_types: List[str],
    splits: List[str],
    max_shard_size: str,
    num_proc: Optional[int] = None,
) -> Dict[Tuple[str, str], str]:
    """Convert a fluid dataset from HDF5 to HF Dataset format."""
    dataset_dir = os.path.join(dataset_root, dataset_name)
    output_files = {}
    
    for dtype in dataset_types:
        data_path = os.path.join(dataset_dir, dtype)
        sub_s = config.sub_s_real if dtype == "real" else config.sub_s_numerical
        is_numerical = dtype == "numerical"
        
        # Load mappings
        try:
            sim_id_mapping, time_id_mapping = load_mapping_files(dataset_dir, dtype)
        except FileNotFoundError as e:
            logging.warning(f"Skipping {dataset_name}/{dtype}: {e}")
            continue
        
        for split in splits:
            sim_ids = list(sim_id_mapping.get(split, []))
            time_ids = list(time_id_mapping.get(split, []))
            
            output_path = os.path.join(output_dir, f"{dtype}_{split}")
            if os.path.exists(output_path):
                logging.info(
                    f"Skipping {dataset_name}/{dtype}/{split}: output already exists at {output_path}"
                )
                output_files[(dtype, split)] = output_path
                continue

            if len(sim_ids) == 0:
                # Write an empty split for consistent on-disk structure and wrapper behavior.
                logging.info(f"Writing EMPTY split: {dataset_name}/{dtype}/{split}")
                features = get_fluid_features(is_numerical)
                empty_data = {k: [] for k in features.keys()}
                dataset = Dataset.from_dict(empty_data, features=features)
                dataset.save_to_disk(output_path, max_shard_size=max_shard_size)
                output_files[(dtype, split)] = output_path
                continue
            
            logging.info(f"Converting {dataset_name}/{dtype}/{split}: {len(sim_ids)} samples")
            
            # Create generator function with closure
            def make_generator(
                dataset_dir=dataset_dir,
                data_path=data_path,
                sim_ids=sim_ids,
                time_ids=time_ids,
                horizon=config.horizon,
                sub_s=sub_s,
                is_numerical=is_numerical,
            ):
                return fluid_generator(
                    dataset_dir=dataset_dir,
                    data_path=data_path,
                    sim_ids=sim_ids,
                    time_ids=time_ids,
                    horizon=horizon,
                    sub_s=sub_s,
                    is_numerical=is_numerical,
                )
            
            # Create HF Dataset from generator
            features = get_fluid_features(is_numerical)
            dataset = Dataset.from_generator(
                make_generator,
                features=features,
                num_proc=num_proc,
            )
            
            # Save to disk with automatic sharding
            dataset.save_to_disk(output_path, max_shard_size=max_shard_size)
            
            output_files[(dtype, split)] = output_path
            logging.info(f"  Saved {len(dataset)} samples to {output_path}")
    
    return output_files


def convert_combustion_to_hf(
    dataset_name: str,
    dataset_root: str,
    output_dir: str,
    config: DatasetConfig,
    dataset_types: List[str],
    splits: List[str],
    max_shard_size: str,
    num_proc: Optional[int] = None,
) -> Dict[Tuple[str, str], str]:
    """Convert a combustion dataset from HDF5 to HF Dataset format."""
    dataset_dir = os.path.join(dataset_root, dataset_name)
    surrogate_path = os.path.join(dataset_dir, config.surrogate_path) if config.surrogate_path else None
    output_files = {}
    
    for dtype in dataset_types:
        data_path = os.path.join(dataset_dir, dtype)
        sub_s = config.sub_s_real if dtype == "real" else config.sub_s_numerical
        is_numerical = dtype == "numerical"
        
        # Load mappings
        try:
            sim_id_mapping, time_id_mapping = load_mapping_files(dataset_dir, dtype)
        except FileNotFoundError as e:
            logging.warning(f"Skipping {dataset_name}/{dtype}: {e}")
            continue
        
        for split in splits:
            sim_ids = list(sim_id_mapping.get(split, []))
            time_ids = list(time_id_mapping.get(split, []))
            
            output_path = os.path.join(output_dir, f"{dtype}_{split}")
            if os.path.exists(output_path):
                logging.info(
                    f"Skipping {dataset_name}/{dtype}/{split}: output already exists at {output_path}"
                )
                output_files[(dtype, split)] = output_path
                continue

            if len(sim_ids) == 0:
                # Write an empty split for consistent on-disk structure and wrapper behavior.
                logging.info(f"Writing EMPTY split: {dataset_name}/{dtype}/{split}")
                features = get_combustion_features(is_numerical)
                empty_data = {k: [] for k in features.keys()}
                dataset = Dataset.from_dict(empty_data, features=features)
                dataset.save_to_disk(output_path, max_shard_size=max_shard_size)
                output_files[(dtype, split)] = output_path
                continue
            
            logging.info(f"Converting {dataset_name}/{dtype}/{split}: {len(sim_ids)} samples")
            
            # Create generator function with closure
            def make_generator(
                dataset_dir=dataset_dir,
                data_path=data_path,
                surrogate_path=surrogate_path,
                sim_ids=sim_ids,
                time_ids=time_ids,
                horizon=config.horizon,
                sub_s=sub_s,
                is_numerical=is_numerical,
            ):
                return combustion_generator(
                    dataset_dir=dataset_dir,
                    data_path=data_path,
                    surrogate_path=surrogate_path,
                    sim_ids=sim_ids,
                    time_ids=time_ids,
                    horizon=horizon,
                    sub_s=sub_s,
                    is_numerical=is_numerical,
                )
            
            # Create HF Dataset from generator
            features = get_combustion_features(is_numerical)
            dataset = Dataset.from_generator(
                make_generator,
                features=features,
                num_proc=num_proc,
            )
            
            # Save to disk with automatic sharding
            dataset.save_to_disk(output_path, max_shard_size=max_shard_size)
            
            output_files[(dtype, split)] = output_path
            logging.info(f"  Saved {len(dataset)} samples to {output_path}")
    
    return output_files


def convert_dataset(
    dataset_name: str,
    dataset_root: str,
    output_dir: Optional[str],
    config: DatasetConfig,
    dataset_types: List[str],
    splits: List[str],
    max_shard_size: str,
    num_proc: Optional[int],
) -> Dict[Tuple[str, str], str]:
    """Convert a dataset from HDF5 to HF Dataset format."""
    if output_dir is None:
        output_dir = os.path.join(dataset_root, dataset_name, "hf_dataset")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Converting dataset: {dataset_name}")
    logging.info(f"  Output: {output_dir}")
    logging.info(f"  Max shard size: {max_shard_size}")
    
    if config.is_combustion:
        return convert_combustion_to_hf(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            output_dir=output_dir,
            config=config,
            dataset_types=dataset_types,
            splits=splits,
            max_shard_size=max_shard_size,
            num_proc=num_proc,
        )
    else:
        return convert_fluid_to_hf(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            output_dir=output_dir,
            config=config,
            dataset_types=dataset_types,
            splits=splits,
            max_shard_size=max_shard_size,
            num_proc=num_proc,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 datasets to HuggingFace Dataset format"
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
        help="Data splits to convert",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="500MB",
        help="Maximum shard size (e.g., '500MB', '1GB')",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override default horizon (in_step + out_step)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes for parallel processing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
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
    
    all_output_files = {}
    
    for name in dataset_names:
        config = DATASET_CONFIGS[name]
        
        # Override horizon if specified
        if args.horizon is not None:
            config.horizon = args.horizon
        
        # Determine output directory
        output_dir = args.output_dir
        if output_dir is None and args.dataset_name != "all":
            output_dir = os.path.join(args.dataset_root, name, "hf_dataset")
        elif output_dir is None:
            output_dir = None  # Will be set per-dataset
        
        try:
            output_files = convert_dataset(
                dataset_name=name,
                dataset_root=args.dataset_root,
                output_dir=output_dir,
                config=config,
                dataset_types=args.dataset_types,
                splits=args.splits,
                max_shard_size=args.max_shard_size,
                num_proc=args.num_proc,
            )
            all_output_files.update({(name, *k): v for k, v in output_files.items()})
        except Exception as e:
            logging.error(f"Error converting {name}: {e}")
            continue
        
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
        
        # Optional: combustion surrogate-train conversion
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
    
    logging.info("=" * 60)
    logging.info("Conversion complete!")
    logging.info("Output directories:")
    for key, path in all_output_files.items():
        logging.info(f"  {key}: {path}")


if __name__ == "__main__":
    main()

