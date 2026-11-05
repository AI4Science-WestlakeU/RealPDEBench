"""Matched missing-observation inputs and source-compatible autoregression.

These utilities do not change model weights, fit statistics, load data or call
an inference provider. Future targets are deliberately absent from this API.
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, label


def make_missing_mask(fluid, fraction, kind, seed, *, require_connected=False):
    fluid = np.asarray(fluid)
    if fluid.ndim != 2 or fluid.dtype != np.bool_ or not fluid.any():
        raise ValueError('fluid must be a nonempty two-dimensional boolean mask')
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError('fraction must be finite and in [0, 1)')
    if kind not in {'scattered', 'compact_patch'}:
        raise ValueError('Unknown mask kind')
    coordinates = np.argwhere(fluid)
    count = round(fraction * len(coordinates))
    missing = np.zeros(fluid.shape, dtype=bool)
    rng = np.random.default_rng(seed)
    if count:
        if kind == 'scattered':
            chosen = rng.choice(len(coordinates), count, replace=False)
        else:
            center = coordinates[rng.integers(len(coordinates))]
            distance = np.abs(coordinates - center)
            chosen = np.lexsort((coordinates[:, 1], coordinates[:, 0],
                                 distance.sum(1), distance.max(1)))[:count]
        missing[tuple(coordinates[chosen].T)] = True
    if require_connected and count and label(missing)[1] != 1:
        raise ValueError('This compact mask is disconnected by the geometry')
    return missing


def fill_history(history, fluid, missing, fill, *, measured_channels=(0, 1)):
    history = np.asarray(history)
    fluid, missing = np.asarray(fluid), np.asarray(missing)
    if history.ndim != 4 or not np.issubdtype(history.dtype, np.floating):
        raise ValueError('history must be a floating [time, height, width, channels] array')
    if not np.isfinite(history).all():
        raise ValueError('history must be finite')
    if fluid.shape != history.shape[1:3] or missing.shape != fluid.shape:
        raise ValueError('Spatial shapes must match')
    if fluid.dtype != np.bool_ or missing.dtype != np.bool_ or np.any(missing & ~fluid):
        raise ValueError('Boolean missing mask must be inside the fluid region')
    if fill not in {'zero', 'nearest_observed'}:
        raise ValueError('Unknown filling method')
    channels = tuple(measured_channels)
    if not channels or len(set(channels)) != len(channels) or any(
            not isinstance(c, int) or c < 0 or c >= history.shape[-1] for c in channels):
        raise ValueError('Unique valid measured-channel indices required')
    output = history.copy()
    if not missing.any():
        return output
    if fill == 'zero':
        for channel in channels:
            output[:, missing, channel] = 0
    else:
        observed = fluid & ~missing
        if not observed.any():
            raise ValueError('Nearest filling needs an observed fluid cell')
        _, indices = distance_transform_edt(~observed, return_indices=True)
        for channel in channels:
            output[:, missing, channel] = history[:, indices[0][missing], indices[1][missing], channel]
    return output


@torch.inference_mode()
def autoregressive_predict(model, normalizer, history, blocks=1, *, device=None):
    """Return physical-unit predictions, preserving official normalization flow.

    history includes all model input channels, including physical-zero initial
    pressure for real cylinder data. Subsequent blocks retain every predicted
    channel, as in the released eval.py. No ground-truth frames are accepted.
    """
    if model.training:
        raise ValueError('Set model.eval() before evaluation')
    if type(blocks) is not int or blocks < 1:
        raise ValueError('blocks must be a positive integer')
    model_device = next((p.device for p in model.parameters()), None)
    if model_device is None:
        model_device = next((b.device for b in model.buffers()), torch.device('cpu'))
    requested = torch.device(device) if device is not None else model_device
    if requested.type not in {'cpu', 'cuda'}:
        raise ValueError('Evaluation supports CPU or CUDA devices')
    if requested.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is not available')
    if requested.type == 'cuda' and requested.index is None:
        requested = torch.device('cuda', torch.cuda.current_device())
    model_tensors = tuple(model.parameters()) + tuple(model.buffers())
    if any(value.device != requested for value in model_tensors):
        raise ValueError('Move the model to the requested device before prediction')
    normalizer_device = torch.device(getattr(normalizer, 'device', requested))
    if normalizer_device.type == 'cuda' and normalizer_device.index is None:
        normalizer_device = torch.device('cuda', torch.cuda.current_device())
    if normalizer_device != requested:
        raise ValueError('Construct the normalizer on the model device')
    tensor = torch.as_tensor(history, dtype=torch.float32, device=requested).contiguous()
    if tensor.ndim != 4 or not torch.isfinite(tensor).all():
        raise ValueError('Finite [time, height, width, channels] history required')
    tensor = tensor.unsqueeze(0)
    normalized, _ = normalizer.preprocess(tensor, tensor)
    predictions = []
    for _ in range(blocks):
        forecast = model(normalized)
        if forecast.device != requested or forecast.shape != normalized.shape:
            raise ValueError('Forecast must retain the device, window shape and channels')
        _, physical = normalizer.postprocess(normalized, forecast)
        if not torch.isfinite(physical).all():
            raise ValueError('Nonfinite model forecast')
        predictions.append(physical)
        normalized, _ = normalizer.preprocess(physical, physical)
    return torch.cat(predictions, dim=1)
