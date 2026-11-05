"""Physical-unit velocity diagnostics; spatial regions never alter targets."""
import numpy as np


def region_metrics(prediction, target, region):
    if not np.any(region): return None
    a = prediction[:, region, :].astype(np.float64)
    b = target[:, region, :].astype(np.float64)
    return {'relative_l2': float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12)),
            'velocity_rmse': float(np.sqrt(np.mean((a-b)**2)))}


def score_velocity(prediction, target, fluid, missing):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError('Matching [time,height,width,channels] predictions and targets required')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Metrics require finite arrays')
    a, b = prediction.astype(np.float64), target.astype(np.float64)
    ae = .5*np.mean((a-a.mean(axis=0, keepdims=True))**2, axis=0).sum(-1)
    be = .5*np.mean((b-b.mean(axis=0, keepdims=True))**2, axis=0).sum(-1)
    return {'all_fluid': region_metrics(a, b, fluid),
            'observed': region_metrics(a, b, fluid & ~missing),
            'occluded': region_metrics(a, b, missing),
            'full_grid_relative_l2': float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12)),
            'fluctuation_ke_mae_full_grid': float(np.mean(np.abs(ae-be))),
            'fluctuation_ke_mae_fluid': float(np.mean(np.abs(ae[fluid]-be[fluid]))),
            'first5_future': region_metrics(a[:5], b[:5], fluid),
            'last5_future': region_metrics(a[-5:], b[-5:], fluid),
            'per_frame_fluid_relative_l2': [region_metrics(a[t:t+1], b[t:t+1], fluid)['relative_l2'] for t in range(len(a))]}
