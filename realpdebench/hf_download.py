from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, Literal, Sequence

import requests
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from realpdebench import __version__

Scenario = Literal["cylinder", "controlled_cylinder", "fsi", "foil", "combustion"]
DatasetType = Literal["real", "numerical"]
Split = Literal["train", "val", "test"]

DEFAULT_HF_DATASET_REPO_ID = "AI4Science-WestlakeU/RealPDEBench"


def _check_version_before_download(
    repo_id: str,
    endpoint: str | None,
    revision: str | None,
    token: str | None,
) -> None:
    """
    Download version.json first and check compatibility BEFORE downloading large files.

    This function fetches the small version.json (~1KB) from HF Hub and checks
    if the current code version meets the minimum required version. If not,
    it raises RuntimeError to prevent wasting bandwidth on incompatible data.

    Version Semantics:
        - data_version: Tracks data content changes (e.g., new trajectories, fixes).
          Incremented for any data update, but does NOT force code upgrades.
        - min_code_version: Minimum code version required to use this data.
          Only incremented for BREAKING changes (format changes, API changes).

    Examples:
        - Adding 6 fsi trajectories: data_version 2.0.0 -> 2.0.1, min_code_version unchanged
        - Fixing data errors: data_version 2.0.1 -> 2.0.2, min_code_version unchanged
        - Format change (V2 -> V3): data_version 3.0.0, min_code_version 0.2.0 -> 0.3.0

    Args:
        repo_id: HF Hub repository ID (e.g., "AI4Science-WestlakeU/RealPDEBench").
        endpoint: Optional HF Hub endpoint (e.g., "https://hf-mirror.com" for China).
        revision: Optional git revision (branch, tag, or commit hash).
        token: Optional HF Hub token for authentication.

    Raises:
        RuntimeError: If code version < min_code_version (breaking incompatibility).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=["version.json"],
                local_dir=tmpdir,
                endpoint=endpoint,
                revision=revision,
                token=token,
            )
        except Exception as e:
            # version.json doesn't exist or network error, skip check
            logging.debug(f"Could not fetch version.json for pre-check: {e}")
            return

        version_file = Path(tmpdir) / "version.json"
        if not version_file.exists():
            return

        try:
            with open(version_file, "r") as f:
                info = json.load(f)
        except (json.JSONDecodeError, IOError):
            return

        min_code = info.get("min_code_version", "0.0.0")
        data_version = info.get("data_version", "unknown")

        def parse_version(v: str) -> tuple:
            return tuple(int(x) for x in v.split(".")[:3])

        try:
            if parse_version(__version__) < parse_version(min_code):
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"DATA VERSION INCOMPATIBLE - DOWNLOAD ABORTED\n"
                    f"{'='*60}\n"
                    f"  HF data version: {data_version}\n"
                    f"  Requires code >= {min_code}\n"
                    f"  Your code version: {__version__}\n"
                    f"\n"
                    f"  Please upgrade first:\n"
                    f"    cd <your-realpdebench-repo>\n"
                    f"    git pull && pip install -e .\n"
                    f"\n"
                    f"  Repo: {info.get('repo_url', '')}\n"
                    f"{'='*60}\n"
                )
        except (ValueError, TypeError):
            pass  # Invalid version format, skip


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
            - "hf_dataset": Arrow shards + index JSONs under `{scenario}/hf_dataset/...`.
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
    # Always allow downloading metadata files (small, helpful).
    patterns.append("README.md")
    patterns.append("version.json")

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
                patterns.append(f"{scenario}/hf_dataset/{dt}/**")
                for sp in splits:
                    patterns.append(f"{scenario}/hf_dataset/{sp}_index_{dt}.json")

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

    # Check version compatibility BEFORE downloading large files
    _check_version_before_download(
        repo_id=repo_id,
        endpoint=endpoint,
        revision=revision,
        token=token,
    )

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
    trajectory_path = dataset_root_path / scenario / "hf_dataset" / dataset_type
    index_path = dataset_root_path / scenario / "hf_dataset" / f"{split}_index_{dataset_type}.json"

    if trajectory_path.exists() and index_path.exists():
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


