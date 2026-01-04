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
from realpdebench.utils.utils import set_seed, add_args_from_config, setup_logging, plot_result
from realpdebench.utils.metrics import eval_metrics, mse_loss, probe_diagnostic


parser = argparse.ArgumentParser(description="Training Configurations")
parser.add_argument("--config", type=str, default="configs/fsi/fno.yaml") 
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--train_data_type", type=str, default="numerical", help="numerical | real")
parser.add_argument("--checkpoint_path", type=str)
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
    # Resolve config path relative to this package if needed (works for `python -m realpdebench.eval`).
    if not os.path.exists(args.config):
        candidate = os.path.join(os.path.dirname(__file__), args.config)
        if os.path.exists(candidate):
            args.config = candidate
    args = add_args_from_config(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_path = os.path.join(args.results_path, args.model_name, args.exp_name+'_eval', f'{current_time}')
    os.makedirs(exp_path, exist_ok=True)

    # Setup logging
    _ = setup_logging(exp_path, is_train=False)
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
        test_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='test',
            N_autoregressive=args.N_autoregressive,
            dataset_type='real',
            **common_kwargs,
        )
        train_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type=args.train_data_type,
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
        test_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='test',
            N_autoregressive=args.N_autoregressive,
            dataset_type='real',
            **common_kwargs,
        )
        train_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type=args.train_data_type,
            mask_prob=args.mask_prob,
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
        test_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='test',
            N_autoregressive=args.N_autoregressive,
            dataset_type='real',
            **common_kwargs,
        )
        train_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type=args.train_data_type,
            mask_prob=args.mask_prob,
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
        test_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='test',
            N_autoregressive=args.N_autoregressive,
            dataset_type='real',
            **common_kwargs,
        )
        train_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type=args.train_data_type,
            mask_prob=args.mask_prob,
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
        test_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='test',
            N_autoregressive=args.N_autoregressive,
            dataset_type='real',
            **common_kwargs,
        )
        train_dataset = DatasetClass(
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            mode='train',
            dataset_type=args.train_data_type,
            mask_prob=args.mask_prob,
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

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                            shuffle=False, pin_memory=True, num_workers=args.num_workers)
    logging.info(f"Data loaded from {test_dataset.dataset_path}")

    # Setup data normalizer
    if args.normalizer == 'none':
        data_normalizer = IdentityNormalizer(device=device)
    elif args.normalizer == 'gaussian':
        data_normalizer = GaussianNormalizer(normalizer_dataset, device=device)
    elif args.normalizer == 'range':
        data_normalizer = RangeNormalizer(normalizer_dataset, device=device)
    else:
        raise ValueError(f"Normalizer {args.normalizer} not supported")
    logging.info(f"Data normalizer: {args.normalizer}")

    # Setup model
    model = load_model(train_dataset, device=device, **vars(args))
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of parameters: {num_params}")

    meta_data = model.load_checkpoint(args.checkpoint_path, device)
    logging.info(f"Checkpoint {args.checkpoint_path} loaded.")

    start_time = time.time()

    logging.info(f"Start testing on {device}")
    model.eval()
    normalized_test_loss = 0.
    
    pred_list, target_list = [], []
    probe_error_list = []
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tqdm.tqdm(test_dataloader)):
            b = input.size(0)
            if 'unmeasured_c' not in locals():
                unmeasured_c = 0
                for c_ in range(target.shape[-1]):
                    if torch.all(target[..., c_] == 0):
                        unmeasured_c += 1
            c = target.shape[-1] - unmeasured_c

            in_control = False
            if input.shape[-1] != target.shape[-1]:
                para_c = input.shape[-1] - target.shape[-1]
                para_input = input[..., -para_c:]
                in_control = True

            input, target = data_normalizer.preprocess(input, target)

            preds = [input]
            for i in range(args.N_autoregressive):
                p = model(preds[-1])
                _, p = data_normalizer.postprocess(preds[-1], p)
                if in_control: p = torch.cat([p, para_input.to(p.device)], dim=-1)
                p, _ = data_normalizer.preprocess(p, target)
                preds.append(p)

            pred = torch.cat(preds[1:], dim=1)
            if in_control: pred = pred[..., :-para_c]
            normalized_test_loss += mse_loss(pred[..., :c], target[..., :c]).reshape(b, -1).mean().item()

            _, pred = data_normalizer.postprocess(input, pred)
            _, target = data_normalizer.postprocess(input, target)

            if batch_idx == 0 and args.N_plot > 0:
                plot_result(pred, target, exp_path, args.N_plot, unmeasured_c)
                # assert False, "Finish plotting"
            
            if hasattr(args, 'probe_diagnostic') and args.probe_diagnostic:
                if batch_idx == 0:
                    probe_error_list_ = probe_diagnostic(pred, target, test_dataset.d, test_dataset.center_x, \
                                    test_dataset.center_y, test_dataset.sub_s_real, N_plot=args.N_plot_probe, \
                                    exp_path=exp_path)
                else:
                    probe_error_list_ = probe_diagnostic(pred, target, test_dataset.d, test_dataset.center_x, \
                                    test_dataset.center_y, test_dataset.sub_s_real)
                probe_error_list.extend(probe_error_list_)

            pred_list.append(pred.cpu())
            target_list.append(target.cpu())

        normalized_test_loss /= len(test_dataloader)
        if args.N_autoregressive > 4:
            eval_batch_size = args.test_batch_size
        else:
            eval_batch_size = torch.cat(pred_list, dim=0).shape[0]
        test_rmse, test_mae, test_rel_l2_error, test_r2, test_ke_error, test_f_error, test_low_f_error, test_mid_f_error, \
            test_high_f_error, test_rel_low_f_error, test_rel_mid_f_error, test_rel_high_f_error, test_freq_error = \
            eval_metrics(torch.cat(pred_list, dim=0), torch.cat(target_list, dim=0), c, eval_batch_size)
        
        logging.info(f"Test results: \n" + \
                    f"normalized mse loss: {normalized_test_loss:.5f}, rmse: {test_rmse:.5f}, mae: {test_mae:.5f}, rel l2 error: " + \
                    f"{test_rel_l2_error:.5f}, r2: {test_r2:.5f}, ke error: {test_ke_error:.5f}, f error: {test_f_error:.5f}, " + \
                    f"low f error: {test_low_f_error:.5f}, mid f error: {test_mid_f_error:.5f}, high f error: {test_high_f_error:.5f}, " + \
                    f"rel low f error: {test_rel_low_f_error:.5f}, rel mid f error: {test_rel_mid_f_error:.5f}, rel high f error: " + \
                    f"{test_rel_high_f_error:.5f}, freq error: {test_freq_error:.5f}")

        if hasattr(args, 'probe_diagnostic') and args.probe_diagnostic:
            probe_error = np.mean(probe_error_list)
            logging.info(f"Probe based diagnostic: {probe_error:.5f}")

    end_time = time.time()
    logging.info(f"Testing complete, time cost is {(end_time-start_time)/60:.2f} min")
    logging.info(f"Results saved at {exp_path}")

