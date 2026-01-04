"""
HF Arrow-backed dataset wrapper for combustion surrogate training.

This module implements a drop-in replacement for `SurrogateDataset` defined in
`data/combustion_surrogate_dataset.py`, but loads from pre-converted HF
Arrow-sharded datasets (via `datasets.load_from_disk`).

It matches the original surrogate behavior:
- `__getitem__` ignores `idx` and samples (sim_id, time_id) with `random.choice`
- input is `numerical_surrogate_train/measured_data` with two extra channels:
  gas_ratio and equivalence_ratio parsed from `sim_id`
- output is `real_surrogate_train/trajectory` with a singleton channel dim
- `__len__` matches the original epoch-sizing logic (mode-dependent only)

Expected HF conversion output (under `{dataset_root}/combustion/hf_dataset/`):
- `surrogate_train/` (HF dataset saved via `save_to_disk`)
- `surrogate_train_sim_ids.txt` (sim_id order used during conversion)
- `surrogate_train_meta.json` (step/n_sim_frame/subsampling used during conversion)
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from typing import Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

from realpdebench.data.dataset import RealDataset


SIM_ID_PATTERN = r"(\d+)NH3_(\d+\.?\d*)\.h5"


class CombustionSurrogateHFDataset(RealDataset):
    """
    HF Arrow-backed surrogate dataset for combustion.

    Args:
        dataset_name: Must be "combustion".
        dataset_root: Root directory containing dataset folders.
        mode: "train" or "test" (only affects __len__ as in original).
        train_ratio: Used only in __len__ for non-train modes (kept for parity).
        step: Window length T returned by __getitem__.
        n_sim_frame: Number of frames used to construct `time_ids = range(n_sim_frame - step)`.
        n_sim_frame_test: Kept for signature parity with the original; unused.
        sub_s_real: Must match conversion settings; kept for signature parity.
        sub_s_numerical: Must match conversion settings; kept for signature parity.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        mode: str,
        hf_auto_download: bool = False,
        hf_repo_id: str = "AI4Science-WestlakeU/RealPDEBench",
        hf_endpoint: str | None = None,
        hf_revision: str | None = None,
        train_ratio: float = 0.8,
        step: int = 20,
        n_sim_frame: int = 40,
        n_sim_frame_test: int = 2001,
        sub_s_real: int = 1,
        sub_s_numerical: int = 1,
    ):
        # Skip RealDataset.__init__ (HDF5 counting); keep Dataset interface.
        Dataset.__init__(self)

        if dataset_name != "combustion":
            raise ValueError(f"CombustionSurrogateHFDataset only supports dataset_name='combustion', got {dataset_name!r}")

        if mode not in {"train", "test"}:
            raise ValueError(f"mode must be 'train' or 'test', got {mode!r}")

        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.mode = mode

        self.train_ratio = float(train_ratio)
        self.step = int(step)
        self.n_sim_frame = int(n_sim_frame)
        self.n_sim_frame_test = int(n_sim_frame_test)  # unused; kept for parity
        self.sub_s_real = int(sub_s_real)
        self.sub_s_numerical = int(sub_s_numerical)

        self.dataset_dir = os.path.join(dataset_root, dataset_name)
        self.hf_dataset_dir = os.path.join(self.dataset_dir, "hf_dataset")

        arrow_path = os.path.join(self.hf_dataset_dir, "surrogate_train")
        if not os.path.exists(arrow_path) and hf_auto_download:
            from realpdebench.hf_download import download_realpdebench

            _ = download_realpdebench(
                dataset_root=dataset_root,
                scenarios=["combustion"],
                what="hf_dataset",
                include_surrogate_train=True,
                repo_id=hf_repo_id,
                endpoint=hf_endpoint,
                revision=hf_revision,
            )

        if not os.path.exists(arrow_path):
            raise FileNotFoundError(
                f"HF Arrow surrogate dataset not found: {arrow_path}\n"
                "To download from HF Hub, run (example):\n"
                f"  realpdebench download --dataset-root {dataset_root} --scenario combustion "
                "--what hf_dataset --include-surrogate-train\n"
                "Or generate it locally from HDF5 via `python -m realpdebench.utils.convert_hdf5_to_hf "
                "--include_surrogate_train ...`."
            )

        # Validate against conversion meta, if present.
        meta_path = os.path.join(self.hf_dataset_dir, "surrogate_train_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            mismatches = []
            if int(meta.get("step", self.step)) != self.step:
                mismatches.append(f"step (meta={meta.get('step')} vs init={self.step})")
            if int(meta.get("n_sim_frame", self.n_sim_frame)) != self.n_sim_frame:
                mismatches.append(f"n_sim_frame (meta={meta.get('n_sim_frame')} vs init={self.n_sim_frame})")
            if int(meta.get("sub_s_real", self.sub_s_real)) != self.sub_s_real:
                mismatches.append(f"sub_s_real (meta={meta.get('sub_s_real')} vs init={self.sub_s_real})")
            if int(meta.get("sub_s_numerical", self.sub_s_numerical)) != self.sub_s_numerical:
                mismatches.append(f"sub_s_numerical (meta={meta.get('sub_s_numerical')} vs init={self.sub_s_numerical})")
            if mismatches:
                raise ValueError(
                    "Surrogate HF dataset meta does not match dataset init args: "
                    + ", ".join(mismatches)
                    + "\nRe-run conversion with matching parameters or instantiate with the meta settings."
                )

        logging.info(f"Loading surrogate HF dataset from Arrow: {arrow_path}")
        self.hf_dataset = load_from_disk(arrow_path)

        # Sim IDs order (for deterministic mapping sim_id -> row range)
        sim_ids_path = os.path.join(self.hf_dataset_dir, "surrogate_train_sim_ids.txt")
        if not os.path.exists(sim_ids_path):
            raise FileNotFoundError(
                f"Missing surrogate sim_id list: {sim_ids_path}\n"
                "Re-run conversion with --include_surrogate_train, or re-download with "
                "`realpdebench download --scenario combustion --what hf_dataset --include-surrogate-train`."
            )
        with open(sim_ids_path, "r") as f:
            self.sim_ids: List[str] = [line.strip() for line in f if line.strip()]

        if self.n_sim_frame <= self.step:
            raise ValueError(f"n_sim_frame={self.n_sim_frame} must be > step={self.step}")

        # Mirrors SurrogateDataset: time_ids = range(n_sim_frame - step)
        self.time_ids: List[int] = list(range(self.n_sim_frame - self.step))

        self.n_sim = len(self.sim_ids)
        self._n_time = len(self.time_ids)

        expected_len = self.n_sim * self._n_time
        if len(self.hf_dataset) != expected_len:
            raise ValueError(
                "Unexpected surrogate HF dataset size.\n"
                f"  len(hf_dataset)={len(self.hf_dataset)}\n"
                f"  expected={expected_len} (= n_sim={self.n_sim} × n_time={self._n_time})\n"
                "This usually means the conversion parameters (step/n_sim_frame) don't match the dataset init args."
            )

        self._sim_id_to_idx: Dict[str, int] = {sid: i for i, sid in enumerate(self.sim_ids)}
        self._time_id_to_idx: Dict[int, int] = {tid: i for i, tid in enumerate(self.time_ids)}

    @staticmethod
    def _decode_array(binary_data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Decode float32 bytes into a numpy array with the given shape."""
        return np.frombuffer(binary_data, dtype=np.float32).reshape(shape)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mirror original: ignore idx, sample randomly.
        sim_id = random.choice(self.sim_ids)
        time_id = random.choice(self.time_ids)

        sim_idx = self._sim_id_to_idx[sim_id]
        time_idx = self._time_id_to_idx[time_id]
        row_idx = sim_idx * self._n_time + time_idx

        row = self.hf_dataset[row_idx]

        # Safety checks to catch ordering mismatches early.
        if row["sim_id"] != sim_id or int(row["time_id"]) != int(time_id):
            raise RuntimeError(
                "HF surrogate dataset ordering mismatch. Expected "
                f"(sim_id={sim_id}, time_id={time_id}) but got "
                f"(sim_id={row['sim_id']}, time_id={row['time_id']}).\n"
                "Re-run conversion; the dataset must be generated with sim_ids outer loop and time_ids inner loop."
            )

        real_shape = (row["real_shape_t"], row["real_shape_h"], row["real_shape_w"])
        numerical_shape = (
            row["numerical_shape_t"],
            row["numerical_shape_h"],
            row["numerical_shape_w"],
            row["numerical_channels"],
        )

        real = self._decode_array(row["real"], real_shape)
        numerical = self._decode_array(row["numerical"], numerical_shape)

        # Match original dtype/shapes
        real_data = torch.tensor(real, dtype=torch.float32).unsqueeze(-1)  # (T,H,W,1)
        numerical_data = torch.tensor(numerical, dtype=torch.float32)  # (T,H,W,C)

        match = re.match(SIM_ID_PATTERN, sim_id)
        if match is None:
            raise ValueError(f"sim_id {sim_id!r} does not match expected pattern {SIM_ID_PATTERN!r}")

        gas_ratio = int(match.group(1))
        equivalence_ratio = float(match.group(2))

        gas_ratio_channel = torch.ones_like(numerical_data[..., [0]]) * gas_ratio
        equivalence_ratio_channel = torch.ones_like(numerical_data[..., [0]]) * equivalence_ratio
        numerical_data = torch.cat([numerical_data, gas_ratio_channel, equivalence_ratio_channel], dim=-1)

        return numerical_data, real_data  # (input, output)

    def __len__(self) -> int:
        # Mirror original epoch-sizing behavior.
        if self.mode == "train":
            return int(self.n_sim * self.n_sim_frame)
        return int(self.n_sim * self.n_sim_frame / self.train_ratio * (1 - self.train_ratio))


