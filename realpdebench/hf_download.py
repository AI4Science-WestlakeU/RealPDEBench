from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Literal, Sequence

import requests
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

Scenario = Literal["cylinder", "controlled_cylinder", "fsi", "foil", "combustion"]
DatasetType = Literal["real", "numerical"]
Split = Literal["train", "val", "test"]

DEFAULT_HF_DATASET_REPO_ID = "AI4Science-WestlakeU/RealPDEBench"
ALL_SCENARIOS: tuple[Scenario, ...] = (
    "cylinder",
    "controlled_cylinder",
    "fsi",
    "foil",
    "combustion",
)


def _dedup_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def build_allow_patterns(
    *,
    scenarios: Sequence[Scenario],
    what: Literal["metadata", "hf_dataset", "all"],
    dataset_types: Sequence[DatasetType] | None = None,
    splits: Sequence[Split] | None = None,
    include_surrogate_train: bool = False,
) -> list[str]:
    """
    Build Hugging Face `allow_patterns` for downloading subsets of RealPDEBench.

    Args:
        scenarios: Scenario folders to include (e.g., "fsi").
        what: Which artifacts to download.
            - "metadata": only JSON metadata under each scenario (test_mode groups).
            - "hf_dataset": only Arrow shards under `{scenario}/hf_dataset/...`.
            - "all": both metadata and hf_dataset.
        dataset_types: Which dataset types to include for hf_dataset ("real", "numerical").
            If None, includes both.
        splits: Which splits to include ("train", "val", "test"). If None, includes all.
        include_surrogate_train: If True, include combustion surrogate-train artifacts
            under `combustion/hf_dataset/surrogate_train` plus its metadata files.

    Returns:
        List of glob patterns suitable for `huggingface_hub.snapshot_download(allow_patterns=...)`.
    """
    if not scenarios:
        raise ValueError("scenarios must be non-empty.")

    if dataset_types is None:
        dataset_types = ("real", "numerical")
    if splits is None:
        splits = ("train", "val", "test")

    patterns: list[str] = []
    # Always allow downloading the dataset card (small, helpful).
    patterns.append("README.md")

    for scenario in scenarios:
        if what in {"metadata", "all"}:
            patterns.extend(
                [
                    f"{scenario}/in_dist_test_params_*.json",
                    f"{scenario}/out_dist_test_params_*.json",
                    f"{scenario}/remain_params_*.json",
                ]
            )
        if what in {"hf_dataset", "all"}:
            for dt in dataset_types:
                for sp in splits:
                    patterns.append(f"{scenario}/hf_dataset/{dt}_{sp}/**")

        if include_surrogate_train:
            if scenario != "combustion":
                continue
            # combustion-only: surrogate model training artifacts
            patterns.extend(
                [
                    "combustion/hf_dataset/surrogate_train/**",
                    "combustion/hf_dataset/surrogate_train_sim_ids.txt",
                    "combustion/hf_dataset/surrogate_train_meta.json",
                ]
            )

    return _dedup_keep_order(patterns)


def download_realpdebench(
    *,
    dataset_root: str | os.PathLike[str],
    scenarios: Sequence[Scenario],
    what: Literal["metadata", "hf_dataset", "all"] = "metadata",
    dataset_types: Sequence[DatasetType] | None = None,
    splits: Sequence[Split] | None = None,
    include_surrogate_train: bool = False,
    repo_id: str = DEFAULT_HF_DATASET_REPO_ID,
    endpoint: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """
    Download RealPDEBench artifacts from Hugging Face dataset repo using `snapshot_download`.

    By default, only downloads metadata JSONs to avoid accidentally pulling large data.

    Returns:
        A dict with keys:
          - "repo_id"
          - "dataset_root" (absolute path)
          - "allow_patterns"
          - "snapshot_path" (only if dry_run is False)
    """
    allow_patterns = build_allow_patterns(
        scenarios=scenarios,
        what=what,
        dataset_types=dataset_types,
        splits=splits,
        include_surrogate_train=include_surrogate_train,
    )

    dataset_root_path = Path(dataset_root).expanduser().resolve()
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    # Recommended behind GFW / when using hf-mirror: disable Xet to avoid extra domains.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    result: dict[str, object] = {
        "repo_id": repo_id,
        "dataset_root": str(dataset_root_path),
        "allow_patterns": allow_patterns,
    }

    if dry_run:
        return result

    # If the caller didn't pass endpoint explicitly, allow env-based override.
    endpoint = endpoint or os.environ.get("HF_ENDPOINT")

    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            local_dir=str(dataset_root_path),
            endpoint=endpoint,
            revision=revision,
            token=token,
        )
    except (HfHubHTTPError, LocalEntryNotFoundError, requests.exceptions.RequestException) as e:
        raise RuntimeError(
            "Failed to download from Hugging Face Hub.\n"
            "Tips:\n"
            "  - If you are behind GFW, try `--endpoint https://hf-mirror.com` (or set env `HF_ENDPOINT`).\n"
            "  - If you hit rate limits (HTTP 429) or need auth (HTTP 401/403), login and set env `HF_TOKEN=...`.\n"
            "  - We recommend setting env `HF_HUB_DISABLE_XET=1` to avoid extra domains.\n"
            f"Details: {type(e).__name__}: {e}"
        ) from e
    result["snapshot_path"] = snapshot_path
    return result


def ensure_hf_artifacts(
    *,
    dataset_root: str | os.PathLike[str],
    scenario: Scenario,
    dataset_type: DatasetType,
    split: Split,
    need_test_params_json: bool,
    hf_auto_download: bool,
    repo_id: str = DEFAULT_HF_DATASET_REPO_ID,
    endpoint: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> None:
    """
    Ensure required HF Arrow artifacts exist locally; optionally auto-download them.

    This is intended to be used by HF Arrow-backed dataset wrappers.
    """
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    arrow_path = dataset_root_path / scenario / "hf_dataset" / f"{dataset_type}_{split}"

    if arrow_path.exists():
        # NOTE: test params jsons are checked separately by dataset wrappers.
        return

    if not hf_auto_download:
        return

    _ = download_realpdebench(
        dataset_root=str(dataset_root_path),
        scenarios=[scenario],
        what="all" if need_test_params_json else "hf_dataset",
        dataset_types=[dataset_type],
        splits=[split],
        include_surrogate_train=False,
        repo_id=repo_id,
        endpoint=endpoint,
        revision=revision,
        token=token,
        dry_run=False,
    )


