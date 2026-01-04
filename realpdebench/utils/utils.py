import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import logging
import matplotlib.pyplot as plt
import tqdm
from torch.utils.tensorboard import SummaryWriter


# config utils
def add_args_from_config(args):
    existing_args = set(vars(args).keys())

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    for key, value in config.items():
        if key not in existing_args:
            setattr(args, key, value)
    return args


# training utils
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(exp_path, is_use_tb=False, is_train=True):
    if is_train:
        log_filename = os.path.join(exp_path, f"training.log")
    else:
        log_filename = os.path.join(exp_path, f"eval.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized at {log_filename}")

    if is_use_tb:
        writer = SummaryWriter(log_dir=exp_path)
        logging.info(f"Tensorboard writer initialized at {writer.log_dir}")
    else:
        writer = None
        
    return writer

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


# test utils
def plot_result(pred, target, exp_path, N_plot, unmeasured_c):
    exp_path = os.path.join(exp_path, "figs")
    os.makedirs(exp_path, exist_ok=True)
    
    b, t_, h, w, c = pred.shape
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    N_plot = min(N_plot, b)
    c = c - unmeasured_c
    for idx in tqdm.tqdm(range(N_plot), desc="Plotting results"):
        for i in range(c):
            fig, axes = plt.subplots(3, 4, figsize=(20, 10))
            for k in range(4):
                t = t_ // 4 * k + (t_ - 1) % 4

                error = np.abs(pred[idx, t, :, :, i] - target[idx, t, :, :, i])
                im1 = axes[0, k].imshow(error)
                axes[0, k].set_title(f"Error, t={t}")
                cbar1 = fig.colorbar(im1, ax=axes[0, k], orientation='vertical', fraction=0.02, pad=0.04)

                im2 = axes[1, k].imshow(pred[idx, t, :, :, i])
                axes[1, k].set_title(f"Prediction, t={t}")        
                cbar1 = fig.colorbar(im2, ax=axes[1, k], orientation='vertical', fraction=0.02, pad=0.04)

                im3 = axes[2, k].imshow(target[idx, t, :, :, i])
                axes[2, k].set_title(f"Ground Truth, t={t}")
                cbar1 = fig.colorbar(im3, ax=axes[2, k], orientation='vertical', fraction=0.02, pad=0.04)

            plt.tight_layout()
            plt.savefig(os.path.join(exp_path, f"pred_target_{idx}_channel{i}.png"))
            plt.close()

    logging.info(f"Visualization results saved at {exp_path}")