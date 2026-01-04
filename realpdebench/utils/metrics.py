
from re import S
import torch
from torch import nn
import math as mt
import numpy as np
import os
import matplotlib.pyplot as plt


def mse_loss(pred, target):
    mse = nn.MSELoss(reduction='none')
    return mse(pred, target)

def kinetic_energy(x):
    '''
    x: torch.Tensor, [b, t, h, w, c]
    Returns: torch.Tensor, [b, h, w]
    '''
    u_prime = ((x[..., 0] - x[..., 0].mean(dim=1, keepdim=True)) ** 2).mean(1)
    v_prime = ((x[..., 1] - x[..., 1].mean(dim=1, keepdim=True)) ** 2).mean(1)
    return 0.5 * (u_prime + v_prime)

def eval_metrics(
    pred, target, c, batch_size=None
):
    """
    pred, target: torch.Tensor, [b, t, h, w, c']
    c: int, number of channels to be evaluated
    Returns: rmse, mae, rel_l2_error, r2, ke_error, f_error, low_f_error, mid_f_error, 
            high_f_error, freq_error
    Adapted from https://github.com/pdebench/PDEBench/blob/main/pdebench/models/metrics.py
    """
    pred_all, target_all = pred[..., :c], target[..., :c]
    b, t, h, w, c = target_all.size()
    device = target.device
    if batch_size is None:
        batch_size = pred.shape[0]

    rmse_list, mae_list, rel_l2_error_list, r2_list, ke_error_list, f_error_list, low_f_error_list, \
        mid_f_error_list, high_f_error_list, rel_low_f_error_list, rel_mid_f_error_list, \
        rel_high_f_error_list, freq_error_list = [], [], [], [], [], [], [], [], [], [], [], [], []
    for i in range(0, pred.shape[0], batch_size):
        pred, target = pred_all[i:i+batch_size], target_all[i:i+batch_size]
        b = pred.shape[0]

        # RMSE
        se = mse_loss(pred, target)
        rmse = torch.sqrt(torch.mean(se))

        # MAE
        mae = torch.mean(torch.abs(pred - target))

        # Relative L2 error
        err_l2 = torch.norm(pred.reshape(b, -1) - target.reshape(b, -1), dim=1)
        norm = torch.norm(target.reshape(b, -1), dim=1)
        rel_l2_error = torch.mean(err_l2 / norm)

        # R2
        r2 = 1 - torch.sum((pred - target) ** 2) / torch.sum((target - target.mean(0, keepdim=True)) ** 2)

        # Kinetic energy error (only for fluid data)
        if c < 2:
            ke_error = 0.
        else:
            pred_ke = kinetic_energy(pred)
            target_ke = kinetic_energy(target)
            ke_error = (pred_ke - target_ke).abs().mean()

        # Fourier space error
        pred_F = torch.fft.fftn(pred, dim=[1, 2, 3])
        target_F = torch.fft.fftn(target, dim=[1, 2, 3])
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros([b, min(t // 2, h // 2, w // 2), c], device=device)
        for i in range(t // 2):
            for j in range(h // 2):
                for k in range(w // 2):
                    it = mt.floor(mt.sqrt(i**2 + j**2 + k**2))
                    if it > min(t // 2, h // 2, w // 2) - 1:
                        continue
                    err_F[:, it] += _err_F[:, i, j, k]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (t * h * w) * 1.0 * 1.0 * 1.0

        iLow = int(np.round(min(t // 2, h // 2, w // 2) / 3))
        iHigh = int(np.round(min(t // 2, h // 2, w // 2) * 2 / 3))
        low_f_error = _err_F[:iLow].mean()
        mid_f_error = _err_F[iLow:iHigh].mean()
        high_f_error = _err_F[iHigh:].mean()
        f_error = _err_F.mean()

        _norm_F = torch.abs(target_F) ** 2
        norm_F = torch.zeros([b, min(t // 2, h // 2, w // 2), c], device=device)
        for i in range(t // 2):
            for j in range(h // 2):
                for k in range(w // 2):
                    it = mt.floor(mt.sqrt(i**2 + j**2 + k**2))
                    if it > min(t // 2, h // 2, w // 2) - 1:
                        continue
                    norm_F[:, it] += _norm_F[:, i, j, k]
        _norm_F = torch.sqrt(torch.mean(norm_F, axis=0)) / (t * h * w) * 1.0 * 1.0 * 1.0

        rel_low_f_error = (_err_F / _norm_F)[:iLow].mean()
        rel_mid_f_error = (_err_F / _norm_F)[iLow:iHigh].mean()
        rel_high_f_error = (_err_F / _norm_F)[iHigh:].mean()

        # Frequency error
        sum_pred = torch.sum(pred, dim=[2, 3, 4])
        sum_target = torch.sum(target, dim=[2, 3, 4])
        sum_pred_F = torch.fft.fftn(sum_pred, dim=1)
        sum_target_F = torch.fft.fftn(sum_target, dim=1)
        freq_error = torch.mean(torch.abs(sum_pred_F - sum_target_F))

        rmse_list.append(rmse)
        mae_list.append(mae)
        rel_l2_error_list.append(rel_l2_error)
        r2_list.append(r2)
        ke_error_list.append(ke_error)
        f_error_list.append(f_error)
        low_f_error_list.append(low_f_error)
        mid_f_error_list.append(mid_f_error)
        high_f_error_list.append(high_f_error)
        rel_low_f_error_list.append(rel_low_f_error)
        rel_mid_f_error_list.append(rel_mid_f_error)
        rel_high_f_error_list.append(rel_high_f_error)
        freq_error_list.append(freq_error)

    return torch.tensor(rmse_list).mean(), torch.tensor(mae_list).mean(), torch.tensor(rel_l2_error_list).mean(), \
        torch.tensor(r2_list).mean(), torch.tensor(ke_error_list).mean(), torch.tensor(f_error_list).mean(), \
        torch.tensor(low_f_error_list).mean(), torch.tensor(mid_f_error_list).mean(), torch.tensor(high_f_error_list).mean(), \
        torch.tensor(rel_low_f_error_list).mean(), torch.tensor(rel_mid_f_error_list).mean(), \
        torch.tensor(rel_high_f_error_list).mean(), torch.tensor(freq_error_list).mean()

def probe_diagnostic(pred, target, d, center_x, center_y, sub_s_real, start_time_pred=0, \
                    start_time_target=0, horizon=None, N_plot=None, exp_path=None):
    """
    pred, target: torch.Tensor, [b, t, h, w, c']
    """
    N_probe = 9
    s1, s2 = pred.shape[2], pred.shape[3]
    if horizon is None:
        horizon = pred.shape[1]
    probe_pred_list, probe_target_list = [], []
    probe_error_list = []

    probe_center_y = int(center_y / sub_s_real)
    interval_y = min(2, int(s1 / (N_probe + 1)))
    probe_y = [probe_center_y + interval_y * j for j in range(-(N_probe-1)//2, N_probe-(N_probe-1)//2)]

    for i in range(4):
        if int((2 * d + center_x) / sub_s_real) < s2:
            interval_x = 1
            probe_x = int(((i + 1) * d + center_x) / sub_s_real)
        else:
            interval_x = 0.5
            probe_x = int((0.5 * (i + 2) * d + center_x) / sub_s_real)

        probe_pred = pred[:, start_time_pred:start_time_pred+horizon, probe_y, probe_x, :] # b, t, N_probe, c
        probe_target = target[:, start_time_target:start_time_target+horizon, probe_y, probe_x, :]
        probe_pred_avg = probe_pred.mean(dim=1) # b, N_probe, c
        probe_target_avg = probe_target.mean(dim=1)
        
        if probe_pred_avg.shape[-1] != probe_target_avg.shape[-1]:
            probe_target_avg = probe_target_avg[..., :probe_pred_avg.shape[-1]]
            
        probe_error = torch.mean(torch.abs(probe_pred_avg - probe_target_avg))
        probe_pred_list.append(probe_pred_avg.cpu().numpy())
        probe_target_list.append(probe_target_avg.cpu().numpy())
        probe_error_list.append(probe_error.cpu().numpy())
    
    # normalize
    for i in range(len(probe_pred_list)):
        probe_pred_list[i] -= probe_target_list[i].min(axis=1, keepdims=True)
        probe_target_list[i] -= probe_target_list[i].min(axis=1, keepdims=True)
        normalizer = probe_target_list[i].max(axis=1, keepdims=True)
        normalizer = np.where(normalizer == 0, 1, normalizer)
        probe_pred_list[i] /= normalizer
        probe_target_list[i] /= normalizer
        probe_pred_list[i] *= 1.5
        probe_target_list[i] *= 1.5

    # plot
    if exp_path is not None and N_plot is not None:
        os.makedirs(f"{exp_path}/probe_diagnostic/", exist_ok=True)

        for idx in range(min(N_plot, pred.shape[0])):
            fig, axes = plt.subplots(1, len(probe_pred_list), figsize=(3*len(probe_pred_list), 6))
            if len(probe_pred_list) == 1:
                axes = [axes]

            for i in range(len(probe_pred_list)):
                y_indices = np.linspace(-1, 1, len(probe_y))
                axes[i].plot(probe_target_list[i][idx, :, 0], y_indices, marker='o', label=f"Target", color='blue')
                axes[i].plot(probe_pred_list[i][idx, :, 0], y_indices, marker='x', label=f"Pred", color='orange')
                axes[i].set_ylabel("$y/D$")
                axes[i].set_xlabel(f"$u/U_0$")
                axes[i].legend()
                if interval_x == 1:
                    axes[i].set_title(f"${i+1}D$")
                else:
                    axes[i].set_title(f"${0.5*(i+2)}D$")

            plt.suptitle(f"Probe Based Diagnostic")
            plt.tight_layout()
            plt.savefig(f"{exp_path}/probe_diagnostic/probe_diagnostic_u_{idx}.pdf")
            plt.close()

            fig, axes = plt.subplots(1, len(probe_pred_list), figsize=(3*len(probe_pred_list), 6))
            if len(probe_pred_list) == 1:
                axes = [axes]

            for i in range(len(probe_pred_list)):
                y_indices = probe_y
                axes[i].plot(probe_target_list[i][idx, :, 1], y_indices, marker='o', label=f"Target", color='blue')
                axes[i].plot(probe_pred_list[i][idx, :, 1], y_indices, marker='x', label=f"Pred", color='orange')
                axes[i].set_ylabel("$y/D$")
                axes[i].set_xlabel(f"$u/U_0$")
                axes[i].legend()
                if interval_x == 1:
                    axes[i].set_title(f"${i+1}D$")
                else:
                    axes[i].set_title(f"${0.5*(i+2)}D$")

            plt.suptitle(f"Probe Based Diagnostic")
            plt.tight_layout()
            plt.savefig(f"{exp_path}/probe_diagnostic/probe_diagnostic_v_{idx}.pdf")
            plt.close()

        print(f"Probe based diagnostic plots saved at {exp_path}/probe_diagnostic")
        
    return probe_error_list

if __name__ == "__main__":
    pred = torch.randn(32, 10, 64, 64, 3)
    target = torch.randn(32, 10, 64, 64, 3)
    rmse, mae, rel_l2_error, r2, ke_error, f_error, low_f_error, mid_f_error, high_f_error, \
            rel_low_f_error, rel_mid_f_error, rel_high_f_error, freq_error = eval_metrics(pred, target, 3)