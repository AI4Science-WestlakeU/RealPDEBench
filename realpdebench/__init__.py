"""
RealPDEBench: A benchmark for complex physical systems with paired real-world and simulated data.
"""

__all__ = ["__version__", "check_data_version"]

__version__ = "0.2.0"


def check_data_version(dataset_root: str) -> None:
    """
    Check if the local data version is compatible with this code version.

    Reads version.json from dataset_root and warns if code upgrade is needed.
    """
    import json
    import logging
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

    # Simple version comparison (works for semver like "0.2.0")
    def parse_version(v: str) -> tuple:
        return tuple(int(x) for x in v.split(".")[:3])

    try:
        if parse_version(__version__) < parse_version(min_code):
            logging.warning(
                f"\n"
                f"{'='*60}\n"
                f"DATA VERSION MISMATCH\n"
                f"{'='*60}\n"
                f"  Data version: {data_version}\n"
                f"  Requires code >= {min_code}\n"
                f"  Your code version: {__version__}\n"
                f"\n"
                f"  Please upgrade: {info.get('upgrade_instructions', 'pip install -U realpdebench')}\n"
                f"  Details: {info.get('repo_url', '')}\n"
                f"{'='*60}\n"
            )
    except (ValueError, TypeError):
        pass  # Invalid version format, skip


