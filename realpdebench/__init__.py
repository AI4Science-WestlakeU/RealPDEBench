"""
RealPDEBench: A benchmark for complex physical systems with paired real-world and simulated data.
"""

__all__ = ["__version__", "check_data_version"]

__version__ = "0.2.0"


def check_data_version(dataset_root: str) -> None:
    """
    Check if the local data version is compatible with this code version.

    This function reads version.json from dataset_root and raises RuntimeError
    if the code version is older than the minimum required version.

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
        dataset_root: Path to the dataset root directory containing version.json.

    Raises:
        RuntimeError: If code version < min_code_version (breaking incompatibility).
    """
    import json
    from pathlib import Path

    version_file = Path(dataset_root) / "version.json"
    if not version_file.exists():
        return  # Old data without version file, skip check

    try:
        with open(version_file, "r") as f:
            info = json.load(f)
    except (json.JSONDecodeError, IOError):
        return  # Corrupted or unreadable, skip

    min_code = info.get("min_code_version", "0.0.0")
    data_version = info.get("data_version", "unknown")

    def parse_version(v: str) -> tuple:
        return tuple(int(x) for x in v.split(".")[:3])

    try:
        if parse_version(__version__) < parse_version(min_code):
            raise RuntimeError(
                f"\n{'='*60}\n"
                f"DATA VERSION INCOMPATIBLE\n"
                f"{'='*60}\n"
                f"  Data version: {data_version}\n"
                f"  Requires code >= {min_code}\n"
                f"  Your code version: {__version__}\n"
                f"\n"
                f"  Please upgrade:\n"
                f"    cd <your-realpdebench-repo>\n"
                f"    git pull && pip install -e .\n"
                f"\n"
                f"  Repo: {info.get('repo_url', '')}\n"
                f"{'='*60}\n"
            )
    except (ValueError, TypeError):
        pass  # Invalid version format, skip


