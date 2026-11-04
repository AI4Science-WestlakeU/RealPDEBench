# Transolver-sigma

Transolver-sigma combines axis-factorized spectral convolutions with physics
slice attention. Four spatiotemporal blocks process the input window, attention
pooling collapses the time dimension, and four spatial blocks predict the next
window. Each block splits 128 channels between spectral and attention branches.
The supplied configurations use 8 attention heads, 32 slices, and temporal/spatial
spectral mode limits of 8/16.

The model accepts `[batch, time, height, width, channels]`. Input and output time
and spatial dimensions must match; input/output channel counts may differ.
Shapes are inferred from the dataset by the model factory. Existing checkpoint
parameter names are preserved. The normal benchmark checkpoint loader performs
strict loading.

## Configurations

Five configurations are provided under `realpdebench/configs/<scenario>/transolver_sigma.yaml`:

| Scenario | Input/output time steps | Grid | Input/output channels |
|---|---|---|---|
| controlled_cylinder | 10 / 10 | 64 × 128 | 5 / 3 |
| fsi | 20 / 20 | 64 × 64 | 3 / 3 |
| foil | 20 / 20 | 64 × 128 | 3 / 3 |
| combustion | 20 / 20 | 64 × 64 | 16 / 16 |
| cylinder | 20 / 20 | 64 × 128 | 3 / 3 |

The supplied configurations target real-world training from scratch: seed 0,
20,000 optimizer updates, batch size 4, and Adam with cosine decay to zero.
The initial learning rate is 0.001 for Cylinder, Controlled Cylinder, FSI and
Foil, and 0.0005 for Combustion. There is no warmup, gradient accumulation or
gradient clipping. The standard trainer validates every 400 updates and selects
the checkpoint with the lowest real-world validation RMSE.

`train_loss` returns squared errors, averaged by the standard training entry
point to obtain MSE. Training, normalization, checkpoint selection and metrics
use the existing benchmark implementations. The temporal mode count is capped
by the available real FFT bins: Controlled Cylinder's 10-frame input uses six
temporal bins even though the configured limit is eight.

## Real-world training results

These are single-seed real-world test results for the supplied architecture and
training settings. All runs completed 20,000 updates; the best validation
checkpoint was evaluated using the benchmark's original-scale metrics.
`fRMSE` is the metric named `f error` in the evaluation log.

| Scenario | Parameters | Initial LR | RMSE ↓ | Relative L2 ↓ | fRMSE ↓ |
|---|---:|---:|---:|---:|---:|
| cylinder | 3,118,628 | 0.001 | 0.08944 | 0.07054 | 0.01267 |
| controlled_cylinder | 3,115,270 | 0.001 | 0.00759 | 0.05268 | 0.00087 |
| fsi | 3,118,628 | 0.001 | 0.00802 | 0.05615 | 0.00064 |
| foil | 3,118,628 | 0.001 | 0.00931 | 0.01512 | 0.00086 |
| combustion | 3,155,496 | 0.0005 | 0.02136 | 0.54474 | 0.00248 |

The learning rates were chosen after exploratory runs; these are not multi-seed
averages. Training budgets and learning rates differ from the supplied U-Net and
DPOT configurations, and this model does not use pretrained weights. These
results therefore do not establish a comparison under equal training budgets.

## Run

Set `dataset_root` and `results_path` in a copy of the relevant configuration.
The dataset root must follow the benchmark layout. With the current official
entry points, numerical training data is also required to construct Gaussian
normalization even when the model trains on real data. Ensure existing
normalization caches correspond to the intended data and preprocessing.

From the repository root, using its installed dependencies:

```bash
python -m realpdebench.train \
  --config realpdebench/configs/cylinder/transolver_sigma.yaml \
  --train_data_type real --use_hf_dataset --gpu 0
```

Evaluate the checkpoint selected by validation RMSE:

```bash
python -m realpdebench.eval \
  --config realpdebench/configs/cylinder/transolver_sigma.yaml \
  --checkpoint_path /path/to/selected/model.pth \
  --train_data_type real --use_hf_dataset --gpu 0
```

GPU and dataset-format flags must be passed on the command line. The supplied
configurations use a single prediction window (`N_autoregressive: 1`).

The same model interface also supports the existing simulated-training and
real-world-finetuning entry points. For simulated training, use
`--train_data_type numerical`. For finetuning, set `checkpoint_path` in a copy of
the configuration to the selected simulated-training checkpoint, then use
`--train_data_type real --is_finetune`. The trainer creates a new optimizer and
scheduler for finetuning; it does not resume their previous states. The learning
rates supplied here have only been evaluated for real-world training from
scratch, not for these other two regimes.

## Contract checks

```bash
python -m unittest discover -s tests -p 'test_transolver_sigma.py' -v
```

These tests cover channel layouts, truncated FFT modes, MSE gradients, factory
registration, checkpoint round trips, incompatible input shapes, and the five
supplied configurations and parameter counts. They require no downloaded data
or pretrained weights.
