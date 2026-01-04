import torch
import logging
import argparse
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

from realpdebench.data.fluid_dataset import FSI, Cylinder, ControlledCylinder, Foil


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
        if int((4 * d + center_x) / sub_s_real) < s2:
            interval_x = 1
            probe_x = int(((i + 1) * d + center_x) / sub_s_real)
        else:
            interval_x = 0.5
            probe_x = int((0.5 * (i + 2) * d + center_x) / sub_s_real)

        probe_pred = pred[:, start_time_pred:start_time_pred+horizon, probe_y, probe_x, :] # b, t, N_probe, c
        probe_target = target[:, start_time_target:start_time_target+horizon, probe_y, probe_x, :]
        probe_pred_avg = probe_pred.mean(dim=1) # b, N_probe, c
        probe_target_avg = probe_target.mean(dim=1)
        probe_error = torch.mean(torch.abs(probe_pred_avg - probe_target_avg))
        probe_pred_list.append(probe_pred_avg.cpu().numpy())
        probe_target_list.append(probe_target_avg.cpu().numpy())
        probe_error_list.append(probe_error.cpu().numpy())
    
    # normalize
    for i in range(len(probe_pred_list)):
        probe_pred_list[i] -= probe_pred_list[i].min(axis=1, keepdims=True)
        probe_target_list[i] -= probe_target_list[i].min(axis=1, keepdims=True)
        normalizer = probe_pred_list[i].max(axis=1, keepdims=True)
        normalizer = np.where(normalizer == 0, 1, normalizer)
        probe_pred_list[i] /= normalizer
        normalizer = probe_target_list[i].max(axis=1, keepdims=True)
        normalizer = np.where(normalizer == 0, 1, normalizer)
        probe_target_list[i] /= normalizer
        probe_pred_list[i] *= 1.5
        probe_target_list[i] *= 1.5
        
    # plot
    if exp_path is not None and N_plot is not None and N_plot != 0:
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

parser = argparse.ArgumentParser(description="Training Configurations")
parser.add_argument("--dataset_root", type=str, default="/wutailin/real_benchmark/")
parser.add_argument("--dataset_name", type=str, default="foil")
parser.add_argument("--is_interval", type=eval, default=False)
parser.add_argument("--horizon", type=int, default=200)
parser.add_argument("--N_plot", type=int, default=0)
parser.add_argument("--exp_path", type=str, default="./results/numerical_real_compare")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    dataset_dir = os.path.join(args.dataset_root, args.dataset_name)
    real_dataset_path = os.path.join(dataset_dir, 'real')
    numerical_dataset_path = os.path.join(dataset_dir, 'numerical')
    exp_path = os.path.join(args.exp_path, args.dataset_name)
    os.makedirs(exp_path, exist_ok=True)

    logging.info(f"Data loaded from {real_dataset_path} and {numerical_dataset_path}")

    # Load datasets
    if args.dataset_name == 'fsi':
        dataset = FSI(dataset_name=args.dataset_name, dataset_root=args.dataset_root, mode='test', \
                dataset_type='real')
    elif args.dataset_name == 'cylinder':
        dataset = Cylinder(dataset_name=args.dataset_name, dataset_root=args.dataset_root, mode='test', \
                dataset_type='real')
    elif args.dataset_name == 'controlled_cylinder':
        dataset = ControlledCylinder(dataset_name=args.dataset_name, dataset_root=args.dataset_root, mode='test', \
                dataset_type='real')
    elif args.dataset_name == 'foil':
        dataset = Foil(dataset_name=args.dataset_name, dataset_root=args.dataset_root, mode='test', \
                dataset_type='real')
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

    numerical_data_list = []
    real_data_list = []
    for idx, file in enumerate(os.listdir(real_dataset_path)):
    # for file in ['11812.h5', '11718.h5', '11625.h5']:
        with h5py.File(os.path.join(real_dataset_path, file), "r") as f:
            u_real = f['measured_data']['u'][:, ::dataset.sub_s_real, ::dataset.sub_s_real] # t, s, s
            v_real = f['measured_data']['v'][:, ::dataset.sub_s_real, ::dataset.sub_s_real]
        real_data = torch.tensor(np.stack([u_real, v_real], axis=-1)[None])

        with h5py.File(os.path.join(numerical_dataset_path, file), "r") as f:
            u_numerical = f['measured_data']['u'][:, ::dataset.sub_s_numerical, ::dataset.sub_s_numerical]
            v_numerical = f['measured_data']['v'][:, ::dataset.sub_s_numerical, ::dataset.sub_s_numerical]
        numerical_data = torch.tensor(np.stack([u_numerical, v_numerical], axis=-1)[None])
        
        if args.is_interval:
            total_norm_real = torch.sqrt(torch.sum(real_data[0] ** 2, dim=(1, 2, 3)))
            start_time_real = int(torch.argmin(total_norm_real[:-args.horizon]))
            total_norm_numerical = torch.sqrt(torch.sum(numerical_data[0] ** 2, dim=(1, 2, 3)))
            start_time_numerical = int(torch.argmin(total_norm_numerical[:-args.horizon]))
        else:
            start_time_numerical, start_time_real = 0, 0
            args.horizon = u_real.shape[1]
    
        numerical_data_list.append(numerical_data)
        real_data_list.append(real_data)
        
    print('Start calculating and plotting probe based diagnostic...')
    numerical_data = torch.cat(numerical_data_list, dim=0)
    real_data = torch.cat(real_data_list, dim=0)
    probe_error_list = probe_diagnostic(numerical_data, real_data, dataset.d, dataset.center_x, dataset.center_y, dataset.sub_s_real, \
                    start_time_pred=start_time_numerical, start_time_target=start_time_real, horizon=args.horizon, N_plot=args.N_plot, \
                    exp_path=exp_path)
                        
    print(f"Probe based diagnostic of numerical and real data on {args.dataset_name}: {np.mean(probe_error_list)}")
    