# Cylinder missing-observation evaluation

This optional evaluation uses the released real-cylinder UNet checkpoint and its
published configuration and Gaussian statistics. It compares complete velocity
history, scattered missing cells and connected compact patches, with physical-zero
and nearest-observed filling. The mask is metadata for filling/scoring; the model
keeps its original three input channels and initially zero pressure channel.

The primary manifest contains one previously frozen 20-input/20-future window for
each of all ten official OOD trajectories. Their IDs are checked against the
released real training IDs before evaluation. The older six-window pilot and its
overlapping training spans are not used as the primary cohort.

## Install

From the repository root, in a clean Python 3.10+ environment:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-robustness.txt
```

For CUDA, install the appropriate CUDA-enabled PyTorch/torchvision build instead
of the CPU command above; see https://pytorch.org/get-started/locally/.
The utility moves input tensors to the model device and requires the normalizer
to be constructed on that same device. It does not silently substitute CPU for
an unavailable requested CUDA device.

## Reproduce the primary OOD evaluation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python scripts/eval_missing_observations.py \
  --device cpu --download --output-dir results/missing_observations_ood
```

The single entry point prepares/checks pinned inputs, creates matched masks,
fills history, loads the official model/configuration/statistics, performs fresh
prediction, computes metrics and writes the report. Downloads are public file
reads, not model API calls. The checkpoint is approximately 92 MB; the ten
velocity windows use approximately 26 MB of verified ranges. Full Arrow shards
are not downloaded or rehashed. Every selected u/v range has a recorded hash.

To use a CUDA device, change `--device cpu` to `--device cuda:0` and select a fresh
output directory. No model weights or normalization statistics are fitted.
CPU/GPU runtime validation must be reported separately on the available hardware.

## Outputs

- `protocol.json`: exact inputs, hashes, device, versions, conditions and overrides.
- `cases.jsonl`: each trajectory/condition with UNet and persistence metrics,
  mask/input/prediction hashes, regional scores and per-frame errors.
- `summary.json`, `summary.csv`, `report.md`: equal-trajectory-weighted summary.
- Optional `predictions/*.npz` when `--save-predictions` is requested.

The default is 50 fresh model cases: ten trajectories times complete plus four
30%-missing geometry/fill conditions. Summary reference values from the previously
reported OOD study are approximately 0.072453 (complete), 0.073702 (nearest-filled
scattered) and 0.183645 (nearest-filled compact) for fluid velocity relative L2.
These are errors, not accuracy percentages. Persistence and zero-fill controls
are included. The report never silently mixes a smoke subset with all ten cases.

`--max-windows 1` is an explicit smoke subset, recorded as 1/10 in the report.
`--seed-offsets 0 1 2` adds controlled mask repetitions, not new trajectories.
`--blocks 5` requests 100 future frames and requires longer complete windows in
a custom verified OOD manifest; the bundled primary windows contain 40 total
frames and deliberately fail that request. The prediction helper itself accepts
history only, retains all predicted channels and converts target-scale outputs
back to input-scale normalization between blocks, matching the released evaluator.

By default use a new output directory. To recover an interrupted run, explicitly
add `--resume`: scientific protocol/checkpoint/config/device must match, and every
reused case's mask and filled-input hashes are checked. `execution.json` separates
new cases from reused ones. Reused records are never counted as new inference.
Input assets may be cached only after their checksums pass. Transient public file
download errors receive four bounded attempts; checksum errors fail immediately.

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests -p test_missing_observation.py -v
PYTHONPATH=. python -m unittest discover -s tests -p test_robustness_reporting.py -v
PYTHONPATH=. python -m unittest discover -s tests -p test_robustness_pipeline.py -v
```

The device tests include a real CUDA numerical test, skipped when CUDA hardware
is absent. CPU checks include two-block normalization/all-channel propagation,
tensor input, invalid devices and explicit normalizer/model device agreement.
The offline pipeline tests use a controlled model with the actual Gaussian
normalizer to check the CLI, future-target isolation, saved outputs and safe
resume. Those tests are software checks, not additional benchmark experiments.

For a hardware CUDA smoke run, use a fresh output directory:

```bash
python scripts/eval_missing_observations.py --device cuda:0 --download \
  --max-windows 1 --output-dir results/missing_observations_cuda_smoke
```

This runs five checkpoint conditions on one OOD trajectory; it is not the full
ten-trajectory evaluation. Remove `--max-windows 1` for the complete CUDA run.

Credit Peiyan Hu et al., RealPDEBench. Retain the project's CC BY-NC 4.0 terms
and source attribution. This optional component changes no default training,
model architecture or benchmark evaluation path.
