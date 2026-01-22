import os, sys
import torch
import torch.nn as nn
import tqdm
import logging
import time, datetime
import numpy as np
import argparse
from realpdebench.data.combustion_dataset import CombustionDataset
from realpdebench.data.combustion_hf_dataset import CombustionHFDataset
from realpdebench.data.fluid_dataset import FSI, Cylinder, ControlledCylinder, Foil
from realpdebench.data.fluid_hf_dataset import (
    CylinderHFDataset,
    FSIHFDataset,
    ControlledCylinderHFDataset,
    FoilHFDataset,
)
from realpdebench.data.data_normalizer import IdentityNormalizer, GaussianNormalizer, RangeNormalizer
from realpdebench.model.load_model import load_model
from realpdebench.utils.utils import set_seed, add_args_from_config, setup_logging, cycle
from realpdebench.utils.metrics import eval_metrics, mse_loss


parser = argparse.ArgumentParser(description="Training Configurations")
parser.add_argument("--config", type=str, default="configs/cylinder/fno.yaml") 
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--train_data_type", type=str, default="numerical", help="numerical | real")
parser.add_argument("--is_finetune", action="store_true", help="enable finetuning mode")
parser.add_argument("--use_hf_dataset", action="store_true", 
                    help="Use HuggingFace Arrow-backed dataset wrapper (loads from `hf_dataset/` via `load_from_disk`).")
parser.add_argument(
    "--hf_auto_download",
    action="store_true",
    help="Auto-download required HF artifacts if missing (only when --use_hf_dataset is set).",
)
parser.add_argument(
    "--hf_repo_id",
    type=str,
    default="AI4Science-WestlakeU/RealPDEBench",
    help="HF dataset repo id (only when --use_hf_dataset is set).",
)
parser.add_argument(
    "--hf_endpoint",
    type=str,
    default=None,
    help="Optional HF endpoint (e.g., https://hf-mirror.com).",
)
parser.add_argument(
    "--hf_revision",
    type=str,
    default=None,
    help="Optional HF revision (branch/tag/commit).",
)

if __name__ == "__main__":
    args = parser.parse_args()
    # Resolve config path relative to this package if needed (works for `python -m realpdebench.train`).
    if not os.path.exists(args.config):
        candidate = os.path.join(os.path.dirname(__file__), args.config)
        if os.path.exists(candidate):
            args.config = candidate
    args = add_args_from_config(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_path = os.path.join(args.results_path, args.model_name, 
                    args.exp_name + '_' + args.train_data_type + '_'+str(args.is_finetune), 
                    current_time)
    os.makedirs(exp_path, exist_ok=True)

    # Setup logging
    writer = setup_logging(exp_path, args.is_use_tb)
    if args.is_use_tb:
        for key, value in vars(args).items():
            writer.add_text(key, str(value), 0)
    logging.info(f'args: {args}')

    # Load datasets
    if args.dataset_name == 'combustion':
        if args.use_hf_dataset:
            DatasetClass = CombustionHFDataset
            logging.info("Using CombustionHFDataset (HF Arrow-backed via `datasets.load_from_disk`)")
        else:
            DatasetClass = CombustionDataset
        common_kwargs = {}
        if args.use_hf_dataset:
            common_kwargs = {
                "hf_auto_download": bool(args.hf_auto_download),
                "hf_repo_id": args.hf_repo_id,
                "hf_endpoint": args.hf_endpoint,
                "hf_revision": args.hf_revision,
            }
        train_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type=args.train_data_type,
            mask_prob=args.mask_prob,
            noise_scale=args.noise_scale,
            **common_kwargs,
        )
        val_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='val',
            dataset_type='real',
            **common_kwargs,
        )
        normalizer_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type='numerical',
            **common_kwargs,
        )
    elif args.dataset_name == 'fsi':
        if args.use_hf_dataset:
            DatasetClass = FSIHFDataset
            logging.info("Using FSIHFDataset (HF Arrow-backed via `datasets.load_from_disk`)")
        else:
            DatasetClass = FSI
        common_kwargs = {}
        if args.use_hf_dataset:
            common_kwargs = {
                "hf_auto_download": bool(args.hf_auto_download),
                "hf_repo_id": args.hf_repo_id,
                "hf_endpoint": args.hf_endpoint,
                "hf_revision": args.hf_revision,
            }
        train_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type=args.train_data_type,
            mask_prob=args.mask_prob,
            noise_scale=args.noise_scale,
            **common_kwargs,
        )
        val_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='val',
            dataset_type='real',
            **common_kwargs,
        )
        normalizer_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type='numerical',
            **common_kwargs,
        )
    elif args.dataset_name == 'cylinder':
        if args.use_hf_dataset:
            DatasetClass = CylinderHFDataset
            logging.info("Using CylinderHFDataset (HF Arrow-backed via `datasets.load_from_disk`)")
        else:
            DatasetClass = Cylinder
        common_kwargs = {}
        if args.use_hf_dataset:
            common_kwargs = {
                "hf_auto_download": bool(args.hf_auto_download),
                "hf_repo_id": args.hf_repo_id,
                "hf_endpoint": args.hf_endpoint,
                "hf_revision": args.hf_revision,
            }
        train_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type=args.train_data_type,
            mask_prob=args.mask_prob,
            noise_scale=args.noise_scale,
            **common_kwargs,
        )
        val_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='val',
            dataset_type='real',
            **common_kwargs,
        )
        normalizer_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type='numerical',
            **common_kwargs,
        )
    elif args.dataset_name == 'controlled_cylinder':
        if args.use_hf_dataset:
            DatasetClass = ControlledCylinderHFDataset
            logging.info("Using ControlledCylinderHFDataset (HF Arrow-backed via `datasets.load_from_disk`)")
        else:
            DatasetClass = ControlledCylinder
        common_kwargs = {}
        if args.use_hf_dataset:
            common_kwargs = {
                "hf_auto_download": bool(args.hf_auto_download),
                "hf_repo_id": args.hf_repo_id,
                "hf_endpoint": args.hf_endpoint,
                "hf_revision": args.hf_revision,
            }
        train_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type=args.train_data_type,
            mask_prob=args.mask_prob,
            noise_scale=args.noise_scale,
            **common_kwargs,
        )
        val_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='val',
            dataset_type='real',
            **common_kwargs,
        )
        normalizer_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type='numerical',
            **common_kwargs,
        )
    elif args.dataset_name == 'foil':
        if args.use_hf_dataset:
            DatasetClass = FoilHFDataset
            logging.info("Using FoilHFDataset (HF Arrow-backed via `datasets.load_from_disk`)")
        else:
            DatasetClass = Foil
        common_kwargs = {}
        if args.use_hf_dataset:
            common_kwargs = {
                "hf_auto_download": bool(args.hf_auto_download),
                "hf_repo_id": args.hf_repo_id,
                "hf_endpoint": args.hf_endpoint,
                "hf_revision": args.hf_revision,
            }
        train_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type=args.train_data_type,
            mask_prob=args.mask_prob,
            noise_scale=args.noise_scale,
            **common_kwargs,
        )
        val_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='val',
            dataset_type='real',
            **common_kwargs,
        )
        normalizer_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type='numerical',
            **common_kwargs,
        )
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

    train_dataloader = cycle(torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, 
                                            shuffle=True, pin_memory=True, num_workers=args.num_workers))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size,
                                            shuffle=False, pin_memory=True, num_workers=args.num_workers)
    logging.info(f"Data loaded from {train_dataset.dataset_path}")

    # Setup data normalizer
    if args.normalizer == 'none':
        data_normalizer = IdentityNormalizer(device=device)
    elif args.normalizer == 'gaussian':
        data_normalizer = GaussianNormalizer(normalizer_dataset, device=device)
    elif args.normalizer == 'range':
        data_normalizer = RangeNormalizer(normalizer_dataset, device=device)
    else:
        raise ValueError(f"Normalizer {args.normalizer} not supported")

    # Setup model
    model = load_model(train_dataset, device=device, **vars(args))
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of parameters: {num_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_update)
    else:
        raise ValueError(f"Scheduler {args.scheduler} not supported")

    # Resume training
    if args.is_finetune:
        meta_data = model.load_checkpoint(args.checkpoint_path, device)
        logging.info(f"Checkpoint {args.checkpoint_path} loaded.")

    start_time = time.time()
    best_iteration = 0

    logging.info(f"Start training on {device}")
    pbar = tqdm.tqdm(range(1, args.num_update + 1))
    best_val_loss = float('inf')
    total_loss = 0.
    count = 0

    all_train_losses = []
    all_val_losses = {
        'normalized_mse': [], 'rmse': [], 'mae': [], 'rel_l2_error': [], 
        'r2': [], 'ke_error': [], 'f_error': [], 'low_f_error': [],
        'mid_f_error': [], 'high_f_error': [], 
        'rel_low_f_error': [], 'rel_mid_f_error': [], 'rel_high_f_error': [],
        'freq_error': [],
    }

    for iteration in pbar:
        model.train()
        
        input, target = next(train_dataloader)
        optimizer.zero_grad()
        input, target = data_normalizer.preprocess(input, target)
        
        loss = model.train_loss(input, target).mean()
        loss.backward()
        if args.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

        optimizer.step() 
        scheduler.step()
        total_loss += loss.item()
        count += 1
        pbar.set_postfix(loss=loss.item())
        
        all_train_losses.append(loss.item())
        
        if args.is_use_tb:
            writer.add_scalar("train_loss", loss.item(), iteration)

        if iteration % int(args.num_update / 50) == 0:
            model.eval()
            normalized_val_loss = 0.

            pred_list, target_list = [], []
            with torch.no_grad():
                for input, target in val_dataloader:
                    b = input.size(0)
                    if 'unmeasured_c' not in locals():
                        unmeasured_c = 0
                        for c_ in range(target.shape[-1]):
                            if torch.all(target[..., c_] == 0):
                                unmeasured_c += 1
                    c = target.shape[-1] - unmeasured_c

                    input, target = data_normalizer.preprocess(input, target)

                    pred = model(input)
                    normalized_val_loss += mse_loss(pred[..., :c], target[..., :c]).reshape(b, -1).mean().item()

                    _, pred = data_normalizer.postprocess(input, pred)
                    _, target = data_normalizer.postprocess(input, target)

                    pred_list.append(pred.cpu())
                    target_list.append(target.cpu())

                normalized_val_loss /= len(val_dataloader)
                val_rmse, val_mae, val_rel_l2_error, val_r2, val_ke_error, val_f_error, val_low_f_error, val_mid_f_error, \
                    val_high_f_error, val_rel_low_f_error, val_rel_mid_f_error, val_rel_high_f_error, val_freq_error = \
                    eval_metrics(torch.cat(pred_list, dim=0), torch.cat(target_list, dim=0), c)
                
                all_val_losses['normalized_mse'].append(normalized_val_loss)
                all_val_losses['rmse'].append(val_rmse)
                all_val_losses['mae'].append(val_mae)
                all_val_losses['rel_l2_error'].append(val_rel_l2_error)
                all_val_losses['r2'].append(val_r2)
                all_val_losses['ke_error'].append(val_ke_error)
                all_val_losses['f_error'].append(val_f_error)
                all_val_losses['low_f_error'].append(val_low_f_error)
                all_val_losses['mid_f_error'].append(val_mid_f_error)
                all_val_losses['high_f_error'].append(val_high_f_error)
                all_val_losses['rel_low_f_error'].append(val_rel_low_f_error)
                all_val_losses['rel_mid_f_error'].append(val_rel_mid_f_error)
                all_val_losses['rel_high_f_error'].append(val_rel_high_f_error)
                all_val_losses['freq_error'].append(val_freq_error)
                
                if val_rmse < best_val_loss:
                    best_iteration = iteration
                    best_val_loss = val_rmse
                
            logging.info(f"\nIteration {iteration}, train loss: {total_loss / count:.5f}")
            logging.info(f"Validation results: \n" + \
                        f"normalized mse loss: {normalized_val_loss:.5f}, rmse: {val_rmse:.5f}, mae: {val_mae:.5f}, rel l2 error: " + \
                        f"{val_rel_l2_error:.5f}, r2: {val_r2:.5f}, ke error: {val_ke_error:.5f}, f error: {val_f_error:.5f}, " + \
                        f"low f error: {val_low_f_error:.5f}, mid f error: {val_mid_f_error:.5f}, high f error: {val_high_f_error:.5f}, " + \
                        f"rel low f error: {val_rel_low_f_error:.5f}, rel mid f error: {val_rel_mid_f_error:.5f}, rel high f error: " + \
                        f"{val_rel_high_f_error:.5f}, freq error: {val_freq_error:.5f}")
            total_loss = 0.
            count = 0

            if args.is_use_tb:
                writer.add_scalar("normalized_val_loss", normalized_val_loss, iteration)
                writer.add_scalar("val_rmse", val_rmse, iteration)
                writer.add_scalar("val_mae", val_mae, iteration)
                writer.add_scalar("val_rel_l2_error", val_rel_l2_error, iteration)
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'train_losses': all_train_losses,
                'val_losses': all_val_losses,
                'iteration': iteration,
                'best_iteration': best_iteration,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, os.path.join(exp_path, f"model_{iteration:04d}.pth"))

    end_time = time.time()
    logging.info(f"Training complete, best iteration is {best_iteration}, time cost is {(end_time-start_time)/60:.2f} min")
    logging.info(f"Results saved at {exp_path}")

    if args.is_use_tb:
        writer.close()
