from __future__ import annotations

import argparse
import json
import sys
import os

from realpdebench import __version__
from realpdebench.hf_download import ALL_SCENARIOS, download_realpdebench


def _add_download_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "download",
        help="Download RealPDEBench datasets/metadata from Hugging Face (pattern-based).",
    )
    p.add_argument(
        "--dataset-root",
        required=True,
        help="Local directory where the HF snapshot files will be materialized.",
    )
    p.add_argument(
        "--scenario",
        action="append",
        choices=list(ALL_SCENARIOS),
        help="Scenario to download. Repeatable. If omitted, you must pass --all.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Download for all scenarios (DANGEROUS if used with --what=hf_dataset/all).",
    )
    p.add_argument(
        "--what",
        default="metadata",
        choices=["metadata", "hf_dataset", "all"],
        help="What to download. Default: metadata (safe).",
    )
    p.add_argument(
        "--dataset-type",
        action="append",
        choices=["real", "numerical"],
        help="Dataset type for hf_dataset download. Repeatable. Default: both.",
    )
    p.add_argument(
        "--split",
        action="append",
        choices=["train", "val", "test"],
        help="Split for hf_dataset download. Repeatable. Default: all.",
    )
    p.add_argument(
        "--include-surrogate-train",
        action="store_true",
        help="Also include combustion surrogate-train artifacts (combustion only).",
    )
    p.add_argument(
        "--repo-id",
        default="AI4Science-WestlakeU/RealPDEBench",
        help="HF dataset repo id (default: AI4Science-WestlakeU/RealPDEBench).",
    )
    p.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="Optional HF endpoint (e.g., https://hf-mirror.com).",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision (branch/tag/commit).",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Optional HF token. You can also set env `HF_TOKEN` instead.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved allow_patterns without downloading.",
    )
    p.set_defaults(func=_cmd_download)


def _cmd_download(args: argparse.Namespace) -> int:
    if args.all:
        scenarios = list(ALL_SCENARIOS)
    else:
        scenarios = args.scenario or []
        if not scenarios:
            raise SystemExit("Please pass at least one --scenario or use --all.")

    dataset_types = args.dataset_type
    splits = args.split

    try:
        result = download_realpdebench(
            dataset_root=args.dataset_root,
            scenarios=scenarios,
            what=args.what,
            dataset_types=dataset_types,
            splits=splits,
            include_surrogate_train=bool(args.include_surrogate_train),
            repo_id=args.repo_id,
            endpoint=args.endpoint,
            revision=args.revision,
            token=args.token,
            dry_run=bool(args.dry_run),
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="realpdebench")
    parser.add_argument(
        "--version",
        action="version",
        version=f"realpdebench {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_download_subcommand(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main(sys.argv[1:])


