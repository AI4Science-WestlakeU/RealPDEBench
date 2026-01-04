import os, sys
import torch
import torch.nn as nn
import tqdm
import logging
import time, datetime
import numpy as np
import argparse
from realpdebench.data.combustion_surrogate_dataset import SurrogateDataset
from realpdebench.data.combustion_surrogate_hf_dataset import CombustionSurrogateHFDataset
from realpdebench.data.data_normalizer import IdentityNormalizer, GaussianNormalizer, RangeNormalizer
from realpdebench.model.load_model import load_model
from realpdebench.utils.utils import set_seed, add_args_from_config, setup_logging, cycle
from realpdebench.utils.metrics import mse_loss


parser = argparse.ArgumentParser(description="Training Configurations")
parser.add_argument("--config", type=str, default="configs/combustion/surrogate_model/unet.yaml") 
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument(
    "--use_hf_dataset",
    action="store_true",
    help="Use HuggingFace Arrow-backed surrogate dataset wrapper (loads from `combustion/hf_dataset/surrogate_train`).",
)
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
    # Resolve config path relative to this package if needed (works for `python -m realpdebench.train_surrogate`).
    if not os.path.exists(args.config):
        candidate = os.path.join(os.path.dirname(__file__), args.config)
        if os.path.exists(candidate):
            args.config = candidate
    args = add_args_from_config(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_path = os.path.join(args.results_path, args.model_name, args.exp_name, current_time)
    os.makedirs(exp_path, exist_ok=True)

    # Setup logging
    writer = setup_logging(exp_path, args.is_use_tb)
    if args.is_use_tb:
        for key, value in vars(args).items():
            writer.add_text(key, str(value), 0)
    logging.info(f'args: {args}')

    # Load datasets
    if args.use_hf_dataset:
        DatasetClass = CombustionSurrogateHFDataset
        common_kwargs = {
            "hf_auto_download": bool(args.hf_auto_download),
            "hf_repo_id": args.hf_repo_id,
            "hf_endpoint": args.hf_endpoint,
            "hf_revision": args.hf_revision,
        }
    else:
        DatasetClass = SurrogateDataset
        common_kwargs = {}

    train_dataset = DatasetClass(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        mode='train',
        **common_kwargs,
    )
    test_dataset = DatasetClass(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        mode='test',
        **common_kwargs,
    )
    normalizer_dataset = DatasetClass(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        mode='train',
        **common_kwargs,
    )

    train_dataloader = cycle(torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, 
                                            shuffle=True, pin_memory=True, num_workers=args.num_workers))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                            shuffle=False, pin_memory=True, num_workers=args.num_workers)
    logging.info(f"Data loaded from {train_dataset.real_dataset_path} and {train_dataset.numerical_dataset_path}")

    # Setup data normalizer
    if args.normalizer == 'none':
        data_normalizer = IdentityNormalizer(device=device)
    elif args.normalizer == 'gaussian':
        data_normalizer = GaussianNormalizer(normalizer_dataset, device=device, is_save=False)
    elif args.normalizer == 'range':
        data_normalizer = RangeNormalizer(normalizer_dataset, device=device, is_save=False)
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

    start_time = time.time()
    best_iteration = 0

    logging.info(f"Start training on {device}")
    pbar = tqdm.tqdm(range(1, args.num_update + 1))
    best_test_loss = float('inf')
    total_loss = 0.
    count = 0

    all_train_losses = []
    all_test_losses = {
        'normalized_mse': [], 'rmse': [], 'mae': [], 'rel_l2_error': [], 
    }

    for iteration in pbar:
        model.train()
        
        input, target = next(train_dataloader)
        optimizer.zero_grad()
        input, target = data_normalizer.preprocess(input, target)
        
        pred = model(input)
        loss = mse_loss(pred, target).mean()
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

        if iteration % 50 == 0:
            model.eval()
            normalized_test_loss = 0.

            pred_list, target_list = [], []
            with torch.no_grad():
                for input, target in test_dataloader:
                    b = input.size(0)

                    input, target = data_normalizer.preprocess(input, target)

                    pred = model(input)
                    normalized_test_loss += mse_loss(pred, target).reshape(b, -1).mean().item()

                    _, pred = data_normalizer.postprocess(input, pred)
                    _, target = data_normalizer.postprocess(input, target)

                    pred_list.append(pred.cpu())
                    target_list.append(target.cpu())

                pred, target = torch.cat(pred_list, dim=0), torch.cat(target_list, dim=0)
                # RMSE
                test_rmse = mse_loss(pred, target)
                test_rmse = torch.sqrt(torch.mean(test_rmse)).item()

                # MAE
                test_mae = torch.mean(torch.abs(pred - target))

                # Relative L2 error
                b = pred.size(0)
                err_l2 = torch.norm(pred.reshape(b, -1) - target.reshape(b, -1), dim=1)
                norm = torch.norm(target.reshape(b, -1), dim=1)
                test_rel_l2_error = torch.mean(err_l2 / norm).item()
                
                normalized_test_loss /= len(test_dataloader)
                all_test_losses['normalized_mse'].append(normalized_test_loss)
                all_test_losses['rmse'].append(test_rmse)
                all_test_losses['mae'].append(test_mae)
                all_test_losses['rel_l2_error'].append(test_rel_l2_error)
                
                if test_rmse < best_test_loss:
                    best_iteration = iteration
                    best_test_loss = test_rmse
                
            logging.info(f"\nIteration {iteration}, train loss: {total_loss / count:.5f}")
            logging.info(f"Validation results: \n" + \
                        f"normalized mse loss: {normalized_test_loss:.5f}, rmse: {test_rmse:.5f}, mae: {test_mae:.5f}, rel l2 error: " + \
                        f"{test_rel_l2_error:.5f}")
            total_loss = 0.
            count = 0

            if args.is_use_tb:
                writer.add_scalar("normalized_test_loss", normalized_test_loss, iteration)
                writer.add_scalar("test_rmse", test_rmse, iteration)
                writer.add_scalar("test_mae", test_mae, iteration)
                writer.add_scalar("test_rel_l2_error", test_rel_l2_error, iteration)
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'train_losses': all_train_losses,
                'test_losses': all_test_losses,
                'iteration': iteration,
                'best_iteration': best_iteration,
                'best_test_loss': best_test_loss
            }
            torch.save(checkpoint, os.path.join(exp_path, f"model_{iteration:04d}.pth"))

    end_time = time.time()
    logging.info(f"Training complete, best iteration is {best_iteration}, time cost is {(end_time-start_time)/60:.2f} min")
    logging.info(f"Results saved at {exp_path}")

    if args.is_use_tb:
        writer.close()
