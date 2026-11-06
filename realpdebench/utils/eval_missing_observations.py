"""Evaluate the released cylinder UNet with matched missing velocity observations.

Run from an installed checkout (all dependencies are in pyproject.toml)::

    python -m realpdebench.utils.eval_missing_observations --output-dir results/missing --download

Use --device cuda for GPU inference; --max-windows 1 for a smoke test;
--seed-offsets 0 1 2 for repeated masks. Longer autoregressive runs use
--blocks 2 or 5 together with --config pointing to correspondingly longer
verified velocity windows (the bundled primary windows contain 40 frames). Add --save-predictions to retain physical-unit forecasts. Outputs include
per-case regional metrics, trajectory-weighted summaries, hashes and settings.
A fresh output directory is required unless --resume is explicitly requested.

The bundled config freezes ten official OOD trajectories, excludes released
training trajectories, and pins checkpoint/statistics and exact u/v byte ranges.
Pass --dataset-root /path/to/data to reuse an existing local CylinderHFDataset;
without it, bounded reads avoid downloading full Arrow shards. Both paths verify
the same pinned u/v bytes before evaluation.
The existing model loader, GaussianNormalizer and kinetic-energy metric are
reused. Pressure starts at physical zero; all model channels are carried into
later blocks. Seeds repeat measurements of each trajectory, not the cohort size.
"""

import argparse
import csv
import hashlib
import itertools
import json
import platform
from pathlib import Path
import time
import urllib.request
import urllib.error

import numpy as np
import torch
import yaml
from scipy.ndimage import label
from realpdebench.data.data_normalizer import GaussianNormalizer
from realpdebench.model.load_model import load_model
from realpdebench.utils.metrics import score_velocity
from realpdebench.utils.missing_observation import (
    make_missing_mask,
    fill_history,
    autoregressive_predict,
)


def retry_read(action):
    """Retry bounded transient public-download errors, never checksum errors."""
    for attempt in range(4):
        try:
            return action()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {
                408,
                429,
                500,
                502,
                503,
                504,
            }:
                raise
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_assets(directory, download=False, assets=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, asset in assets.items():
        if Path(name).name != name:
            raise ValueError("Asset filenames must be local basenames")
        url, expected = asset["url"], asset["sha256"]
        path = directory / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"{path}; use --download to fetch pinned public assets")
            temporary = path.with_suffix(path.suffix + ".part")

            def acquire():
                with (
                    urllib.request.urlopen(url, timeout=120) as response,
                    temporary.open("wb") as f,
                ):
                    while chunk := response.read(1024 * 1024):
                        f.write(chunk)

            retry_read(acquire)
            if sha256(temporary) != expected:
                raise ValueError(f"Asset checksum mismatch: {name}")
            temporary.replace(path)
        if sha256(path) != expected:
            raise ValueError(f"Asset checksum mismatch: {name}")


def load_velocity(record, directory, download=False, dataset_revision=None, dataset=None):
    name = record["window_file"]
    if Path(name).name != name:
        raise ValueError("Window filenames must be local basenames")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    ranges = record.get("range_hashes", [])
    if dataset is not None:
        indices = [
            i
            for i, item in enumerate(dataset._indices)
            if item["sim_id"] == record["sim_id"] and item["time_id"] == record["time_id"]
        ]
        if len(indices) != 1:
            raise ValueError("Expected exactly one frozen window in the local HF test index")
        history, target = dataset[indices[0]]
        velocity = torch.cat([history[..., :2], target[..., :2]], dim=0).numpy()
    elif not path.exists():
        if not download:
            raise FileNotFoundError(f"{path}; use --download to fetch pinned window ranges")
        if not dataset_revision or len(ranges) != 2:
            raise ValueError("Bounded download requires the pinned u/v range manifest")
        shard = record["shard"]
        if not shard.startswith("cylinder/hf_dataset/real/") or ".." in Path(shard).parts:
            raise ValueError("Unexpected shard path")
        columns = {}
        for field in ranges:
            if field["column"] not in {"u", "v"}:
                raise ValueError("Expected measured velocity channels")
            start, end = int(field["start"]), int(field["end"])
            url = f"https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench/resolve/{dataset_revision}/{shard}"
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})

            def acquire():
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or not response.headers.get(
                        "Content-Range", ""
                    ).startswith(f"bytes {start}-{end}/"):
                        raise ValueError("HTTP range response did not match requested boundaries")
                    return response.read(end - start + 2)

            raw = retry_read(acquire)
            if len(raw) != end - start + 1 or hashlib.sha256(raw).hexdigest() != field["sha256"]:
                raise ValueError("Velocity range checksum mismatch")
            columns[field["column"]] = np.frombuffer(raw, dtype="<f4").reshape(-1, 64, 128)
        velocity = np.stack([columns["u"], columns["v"]], axis=-1)
        np.savez_compressed(path, velocity=velocity)
    if dataset is None:
        with np.load(path, allow_pickle=False) as archive:
            velocity = archive["velocity"].copy()
    if velocity.ndim != 4 or velocity.shape[1:] != (64, 128, 2) or not np.isfinite(velocity).all():
        raise ValueError("Expected finite cylinder velocity [time,64,128,2]")
    for field in ranges:
        channel = {"u": 0, "v": 1}[field["column"]]
        raw = np.ascontiguousarray(velocity[..., channel], dtype="<f4").tobytes()
        if hashlib.sha256(raw).hexdigest() != field["sha256"]:
            raise ValueError("Cached velocity differs from pinned range")
    return velocity


class WindowShapes:
    def __getitem__(self, index):
        return torch.zeros(20, 64, 128, 3), torch.zeros(20, 64, 128, 3)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = Path.cwd()
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/cylinder/missing_observation_ood.json",
        help="Single OOD cohort, pinned assets and byte-range configuration",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        help="Reuse local cylinder HF Arrow data with the existing dataset loader; never auto-download shards",
    )
    p.add_argument("--assets-dir", type=Path, default=root / ".robustness_cache/assets")
    p.add_argument("--windows-dir", type=Path, default=root / ".robustness_cache/windows")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    p.add_argument(
        "--blocks",
        type=int,
        default=1,
        help="20-frame prediction blocks; default is the primary OOD horizon",
    )
    p.add_argument("--missing-fractions", nargs="+", type=float, default=[0.3])
    p.add_argument("--seed-offsets", nargs="+", type=int, default=[0])
    p.add_argument(
        "--fills",
        nargs="+",
        choices=["zero", "nearest_observed"],
        default=["zero", "nearest_observed"],
    )
    p.add_argument(
        "--max-windows",
        type=int,
        help="Explicit smoke-test subset; recorded in protocol",
    )
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--download",
        action="store_true",
        help="Fetch missing pinned public assets and exact velocity ranges",
    )
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Explicitly reuse completed cases only after protocol and input-hash checks",
    )
    return p.parse_args(argv)


def report(output, rows, protocol):
    groups = {}
    for row in rows:
        c = row["case"]
        key = (c["missing_fraction"], c["mask_kind"], c["fill"])
        for model in ["unet", "persistence"]:
            groups.setdefault((model,) + key, []).append(row)
    summaries = []
    for (model, fraction, kind, fill), group in sorted(groups.items()):
        per_sim = {}
        for row in group:
            per_sim.setdefault(row["case"]["sim_id"], []).append(
                row[model]["all_fluid"]["relative_l2"]
            )
        summaries.append(
            {
                "model": model,
                "missing_fraction": fraction,
                "mask_kind": kind,
                "fill": fill,
                "distinct_trajectories": len(per_sim),
                "cases": len(group),
                "mean_fluid_relative_l2": float(np.mean([np.mean(v) for v in per_sim.values()])),
            }
        )
    result = {
        "complete": True,
        "case_count": len(rows),
        "unet_forward_passes": len(rows) * protocol["blocks"],
        "primary_cohort": protocol["cohort"],
        "device": protocol["device"],
        "summary": summaries,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    with (output / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    lines = [
        "# Cylinder missing-observation evaluation",
        "",
        f"Cohort: {protocol['cohort']}; selected trajectories: {protocol['selected_windows']}/{protocol['available_windows']}; device: {protocol['device']}.",
        "",
        "| Model | Missing | Geometry | Fill | Trajectories | Mean fluid relative L2 |",
        "|---|---:|---|---|---:|---:|",
    ]
    for r in summaries:
        lines.append(
            f"| {r['model']} | {r['missing_fraction']} | {r['mask_kind']} | {r['fill']} | {r['distinct_trajectories']} | {r['mean_fluid_relative_l2']:.6f} |"
        )
    lines += [
        "",
        "Per-trajectory and per-region measurements are in cases.jsonl. Each seed is a repeated measurement, not another independent trajectory.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.blocks < 1 or args.threads < 1 or args.max_windows is not None and args.max_windows < 1:
        raise ValueError("Positive blocks, threads and optional max-windows required")
    if any(not 0 < r < 1 for r in args.missing_fractions):
        raise ValueError("Missing fractions must be in (0,1)")
    if any(len(v) != len(set(v)) for v in [args.missing_fractions, args.seed_offsets, args.fills]):
        raise ValueError("Repeated conditions are not independent cases")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no silent CPU substitution")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    cohort = json.loads(args.config.read_text())
    windows = cohort["windows"]
    ood_ids, train_ids = set(cohort["official_ood_ids"]), set(cohort["released_train_sim_ids"])
    if (
        len({w["sim_id"] for w in windows}) != len(windows)
        or {w["sim_id"] for w in windows} != ood_ids
    ):
        raise ValueError("Manifest must contain every official OOD trajectory exactly once")
    if ood_ids & train_ids:
        raise ValueError("OOD and released training trajectory IDs overlap")
    selected = windows[: args.max_windows] if args.max_windows else windows
    existing_output = args.output_dir.exists() and any(args.output_dir.iterdir())
    if existing_output and not args.resume:
        raise FileExistsError("Use a fresh output directory or explicit --resume")
    prepare_assets(args.assets_dir, args.download, cohort["assets"])
    config = yaml.safe_load((args.assets_dir / "unet_config.yaml").read_text())
    if config["dataset_name"] != "cylinder" or config["model_name"] != "unet":
        raise ValueError("This runner evaluates released cylinder UNet")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    model = load_model(WindowShapes(), device=device, **config)
    state = torch.load(args.assets_dir / "unet_real.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    normalizer = GaussianNormalizer(
        type("PublishedStats", (), {"dataset_dir": str(args.assets_dir)})(), device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "cohort": "official OOD trajectories absent from released real training index",
        "available_windows": len(windows),
        "selected_windows": len(selected),
        "cohort_config_sha256": sha256(args.config),
        "device": str(device),
        "blocks": args.blocks,
        "frames_per_block": 20,
        "missing_fractions": args.missing_fractions,
        "seed_offsets": args.seed_offsets,
        "fills": args.fills,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "checkpoint_sha256": sha256(args.assets_dir / "unet_real.pth"),
        "statistics_sha256": sha256(args.assets_dir / "mean_std.pt"),
        "config_sha256": sha256(args.assets_dir / "unet_config.yaml"),
        "model_revision": cohort["model_revision"],
        "dataset_revision": cohort["dataset_revision"],
        "config_overrides": {
            "device": str(device),
            "batch_size": 1,
            "N_autoregressive": args.blocks,
            "threads": args.threads,
        },
        "reference_yaml_blocks": config["N_autoregressive"],
        "normalization": "published Gaussian statistics; no test-time fitting",
        "geometry": "64x128 cylinder; fluid excludes radius-9 obstacle centered at (32,32)",
        "mask_usage": "same static spatial mask on both measured velocity channels; pressure remains physical zero initially",
        "report_weighting": "mean over seeds within trajectory, then equal mean over trajectories",
        "new_training": False,
        "paid_calls": 0,
        "cache_policy": "Fresh inference by default; explicit resume validates scientific protocol and each filled input",
    }
    cached = {}
    if existing_output:
        previous = json.loads((args.output_dir / "protocol.json").read_text())
        if {k: v for k, v in previous.items() if k != "cache_policy"} != {
            k: v for k, v in protocol.items() if k != "cache_policy"
        }:
            raise ValueError(
                "Resume protocol does not match the stored scientific inputs and settings"
            )
        cases_file = args.output_dir / "cases.jsonl"
        if cases_file.exists():
            for line in cases_file.read_text().splitlines():
                row = json.loads(line)
                cid = row["case"]["case_id"]
                if cid in cached:
                    raise ValueError("Duplicate cached case ID")
                cached[cid] = row
    execution = {
        "resume_requested": args.resume,
        "cached_cases_at_start": len(cached),
        "reused_cached_cases": 0,
        "new_cases": 0,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    dataset = None
    if args.dataset_root is not None:
        from realpdebench.data.fluid_hf_dataset import CylinderHFDataset

        dataset = CylinderHFDataset(
            dataset_name="cylinder",
            dataset_root=str(args.dataset_root),
            dataset_type="real",
            mode="test",
            hf_auto_download=False,
            test_mode="all",
            in_step=20,
            out_step=20,
            N_autoregressive=args.blocks,
            sub_s_real=1,
            noise_scale=0.0,
        )
    rows = []
    for ordinal, window in enumerate(selected):
        velocity = load_velocity(
            window, args.windows_dir, args.download, cohort["dataset_revision"], dataset
        )
        horizon = 20 * args.blocks
        if len(velocity) < 20 + horizon:
            raise ValueError("Window does not contain the requested complete future horizon")
        history = np.concatenate([velocity[:20], np.zeros((20, 64, 128, 1), np.float32)], axis=-1)
        target = velocity[20 : 20 + horizon]
        yy, xx = np.ogrid[:64, :128]
        fluid = (yy - 32) ** 2 + (xx - 32) ** 2 > 81
        conditions = [(0.0, "scattered", "zero", window["mask_seed"])]
        conditions += [
            (fraction, kind, fill, window["mask_seed"] + offset)
            for fraction, kind, fill, offset in itertools.product(
                args.missing_fractions,
                ["scattered", "compact_patch"],
                args.fills,
                args.seed_offsets,
            )
        ]
        for fraction, kind, fill, seed in conditions:
            started = time.perf_counter()
            missing = make_missing_mask(
                fluid, fraction, kind, seed, require_connected=kind == "compact_patch"
            )
            filled = fill_history(history, fluid, missing, fill)
            case_id = f"ood{ordinal:02d}_case{len(rows):04d}"
            case = {
                "case_id": case_id,
                "sim_id": window["sim_id"],
                "time_id": window["time_id"],
                "missing_fraction": fraction,
                "mask_kind": kind,
                "fill": fill,
                "mask_seed": seed,
            }
            if case_id in cached:
                old = cached[case_id]
                if (
                    old["case"] != case
                    or old["input_sha256"] != hashlib.sha256(filled.tobytes()).hexdigest()
                    or old["mask_sha256"] != hashlib.sha256(missing.tobytes()).hexdigest()
                ):
                    raise ValueError("Cached case differs from reconstructed current input")
                if (
                    args.save_predictions
                    and not (args.output_dir / "predictions" / (case_id + ".npz")).exists()
                ):
                    raise ValueError(
                        "Cached predictions were not saved; use a fresh run to request prediction files"
                    )
                rows.append(old)
                execution["reused_cached_cases"] += 1
                print(f"{len(rows)} {case_id} REUSED after explicit checks", flush=True)
                continue
            forecast = autoregressive_predict(model, normalizer, filled, args.blocks, device=device)
            prediction = forecast[0, ..., :2].detach().cpu().numpy().copy()
            persistence = np.repeat(filled[-1:, ..., :2], horizon, axis=0)
            row = {
                "case": case,
                "missing_cells": int(missing.sum()),
                "fluid_cells": int(fluid.sum()),
                "mask_components_4connected": int(label(missing)[1]),
                "mask_sha256": hashlib.sha256(missing.tobytes()).hexdigest(),
                "input_sha256": hashlib.sha256(filled.tobytes()).hexdigest(),
                "prediction_array_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
                "unet": score_velocity(prediction, target, fluid, missing),
                "persistence": score_velocity(persistence, target, fluid, missing),
                "seconds_including_metrics": time.perf_counter() - started,
            }
            if args.save_predictions:
                folder = args.output_dir / "predictions"
                folder.mkdir(exist_ok=True)
                np.savez_compressed(folder / (case_id + ".npz"), prediction=prediction)
            rows.append(row)
            execution["new_cases"] += 1
            with (args.output_dir / "cases.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            print(
                f"{len(rows)} {case_id} UNet relative L2={row['unet']['all_fluid']['relative_l2']:.6f}",
                flush=True,
            )
    result = report(args.output_dir, rows, protocol)
    if execution["reused_cached_cases"] != len(cached):
        raise ValueError("Cached output contains unexpected extra cases")
    result.update(execution)
    result["new_forward_passes_this_invocation"] = execution["new_cases"] * args.blocks
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (args.output_dir / "execution.json").write_text(json.dumps(execution, indent=2))
    with (args.output_dir / "execution_history.jsonl").open("a") as f:
        f.write(json.dumps(execution) + "\n")
    print(
        json.dumps({"completed_cases": len(rows), "output_dir": str(args.output_dir)}),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
