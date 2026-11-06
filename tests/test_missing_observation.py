"""Mask, metric and offline runner validation. Run with unittest discovery."""

from realpdebench import eval_missing_observations as cli
import unittest
import numpy as np
import torch
from realpdebench.utils.missing_observation import (
    make_missing_mask,
    fill_history,
    autoregressive_predict,
)


class MaskAndRolloutTests(unittest.TestCase):
    def setUp(self):
        self.fluid = np.ones((12, 16), bool)
        self.history = np.ones((3, 12, 16, 3), np.float32)
        self.history[..., 2] = 7

    def test_exact_matching_counts(self):
        for kind in ["scattered", "compact_patch"]:
            m = make_missing_mask(self.fluid, 0.3, kind, 7)
            self.assertEqual(m.sum(), round(0.3 * self.fluid.sum()))

    def test_constant_nearest_and_pressure_preserved(self):
        m = make_missing_mask(self.fluid, 0.3, "compact_patch", 7)
        a = fill_history(self.history, self.fluid, m, "nearest_observed")
        self.assertTrue(np.array_equal(a, self.history))

    def test_zero_observed_unchanged_and_input_unmodified(self):
        m = make_missing_mask(self.fluid, 0.3, "scattered", 7)
        a = fill_history(self.history, self.fluid, m, "zero")
        self.assertTrue(np.array_equal(a[:, ~m], self.history[:, ~m]))
        self.assertTrue(np.all(self.history[..., :2] == 1))
        self.assertTrue(np.all(a[..., 2] == 7))

    def test_no_missing_copy(self):
        a = fill_history(self.history, self.fluid, np.zeros_like(self.fluid), "zero")
        self.assertTrue(np.array_equal(a, self.history))
        self.assertIsNot(a, self.history)

    def test_bad_fraction(self):
        with self.assertRaises(ValueError):
            make_missing_mask(self.fluid, 1.0, "scattered", 0)

    def test_unknown_fill(self):
        with self.assertRaises(ValueError):
            fill_history(self.history, self.fluid, np.zeros_like(self.fluid), "invented")

    def test_obstacle_missing_rejected(self):
        self.fluid[0, 0] = False
        m = np.zeros_like(self.fluid)
        m[0, 0] = True
        with self.assertRaises(ValueError):
            fill_history(self.history, self.fluid, m, "zero")

    def test_two_block_normalization_and_all_channels(self):
        class Norm:
            mi = torch.tensor([1.0, 2.0, 3.0])
            si = torch.tensor([2.0, 4.0, 8.0])
            mt = torch.tensor([5.0, 6.0, 7.0])
            st = torch.tensor([3.0, 6.0, 9.0])

            def preprocess(self, x, y):
                return (x - self.mi) / self.si, (y - self.mt) / self.st

            def postprocess(self, x, y):
                return x * self.si + self.mi, y * self.st + self.mt

        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 0.25

        model = Model().eval()
        a = autoregressive_predict(model, Norm(), np.zeros((1, 2, 2, 3), np.float32), blocks=2)
        expected = torch.tensor([[4.25, 4.5, 5.875], [10.625, 11.25, 12.484375]])
        self.assertTrue(torch.equal(a[0, :, 0, 0], expected))

    def test_training_mode_rejected(self):
        with self.assertRaises(ValueError):
            autoregressive_predict(torch.nn.Identity(), None, self.history)

    def test_invalid_block_count(self):
        with self.assertRaises(ValueError):
            autoregressive_predict(torch.nn.Identity().eval(), None, self.history, blocks=0)

    def test_cpu_tensor_history_and_device(self):
        class Norm:
            device = "cpu"

            def preprocess(self, x, y):
                return x.to(self.device), y.to(self.device)

            def postprocess(self, x, y):
                return x.to(self.device), y.to(self.device)

        model = torch.nn.Identity().eval()
        actual = autoregressive_predict(
            model, Norm(), torch.from_numpy(self.history), blocks=2, device="cpu"
        )
        self.assertEqual(actual.device.type, "cpu")
        self.assertTrue(
            torch.equal(
                actual,
                torch.from_numpy(np.concatenate([self.history, self.history]))[None],
            )
        )

    def test_normalizer_device_mismatch_rejected(self):
        class Norm:
            device = "meta"

        with self.assertRaisesRegex(ValueError, "normalizer"):
            autoregressive_predict(torch.nn.Identity().eval(), Norm(), self.history, device="cpu")

    def test_unknown_device_rejected(self):
        with self.assertRaises(ValueError):
            autoregressive_predict(torch.nn.Identity().eval(), None, self.history, device="meta")

    def test_boolean_blocks_rejected(self):
        with self.assertRaises(ValueError):
            autoregressive_predict(torch.nn.Identity().eval(), None, self.history, blocks=True)

    def test_nonfinite_input_rejected(self):
        x = self.history.copy()
        x[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            autoregressive_predict(torch.nn.Identity().eval(), None, x)

    def test_buffer_only_model_device_mismatch_rejected_before_forward(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("offset", torch.empty((), device="meta"))

            def forward(self, x):
                raise AssertionError("Device mismatch must fail before forward")

        with self.assertRaisesRegex(ValueError, "Move the model"):
            autoregressive_predict(Model().eval(), None, self.history, device="cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable")
    def test_cuda_prediction_retains_device_and_values(self):
        class Norm:
            device = torch.device("cuda", torch.cuda.current_device())

            def preprocess(self, x, y):
                return x.to(self.device), y.to(self.device)

            def postprocess(self, x, y):
                return x.to(self.device), y.to(self.device)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("offset", torch.tensor(0.25))

            def forward(self, x):
                return x + self.offset

        model = Model().cuda().eval()
        output = autoregressive_predict(model, Norm(), self.history, blocks=2, device="cuda")
        expected = np.concatenate([self.history + 0.25, self.history + 0.5])[None]
        self.assertEqual(output.device.type, "cuda")
        np.testing.assert_allclose(output.cpu().numpy(), expected)


from pathlib import Path
import tempfile
from unittest.mock import patch
import urllib.error
from realpdebench.utils.metrics import score_velocity
from realpdebench.eval_missing_observations import retry_read


class ReportingTests(unittest.TestCase):
    def test_metrics_exact_forecast_and_empty_occlusion(self):
        x = np.ones((20, 4, 5, 2), np.float32)
        fluid = np.ones((4, 5), bool)
        m = score_velocity(x, x, fluid, np.zeros_like(fluid))
        self.assertEqual(m["all_fluid"]["relative_l2"], 0)
        self.assertIsNone(m["occluded"])
        self.assertEqual(m["fluctuation_ke_mae_full_grid"], 0)

    def test_metric_spatial_regions_are_scored_separately(self):
        target = np.ones((20, 4, 5, 2), np.float32)
        pred = target.copy()
        pred[:, 0, 0] = 0
        missing = np.zeros((4, 5), bool)
        missing[0, 0] = True
        m = score_velocity(pred, target, np.ones_like(missing), missing)
        self.assertEqual(m["observed"]["relative_l2"], 0)
        self.assertEqual(m["occluded"]["relative_l2"], 1)

    def test_report_weights_trajectories_not_number_of_seeds(self):
        rows = []
        for sim, error in [("a", 1.0), ("a", 3.0), ("b", 8.0)]:
            metric = {"all_fluid": {"relative_l2": error}}
            rows.append(
                {
                    "case": {
                        "sim_id": sim,
                        "missing_fraction": 0.3,
                        "mask_kind": "scattered",
                        "fill": "zero",
                    },
                    "unet": metric,
                    "persistence": metric,
                }
            )
        protocol = {
            "cohort": "controlled test",
            "blocks": 1,
            "device": "cpu",
            "selected_windows": 2,
            "available_windows": 2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = cli.report(Path(tmp), rows, protocol)
            for item in result["summary"]:
                self.assertEqual(item["mean_fluid_relative_l2"], 5.0)
            for name in ["summary.json", "summary.csv", "report.md"]:
                self.assertTrue((Path(tmp) / name).exists())

    def test_primary_default_is_explicit_one_block(self):
        args = cli.parse_args(["--output-dir", "unused"])
        self.assertEqual(args.blocks, 1)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.seed_offsets, [0])

    def test_existing_outputs_require_explicit_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "existing.txt").write_text("not a fresh run")
            with self.assertRaises(FileExistsError):
                cli.main(["--output-dir", tmp])

    def test_transient_download_error_is_retried(self):
        calls = []

        def read():
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.HTTPError("https://example.invalid", 503, "temporary", {}, None)
            return b"verified later"

        with patch("realpdebench.eval_missing_observations.time.sleep"):
            self.assertEqual(retry_read(read), b"verified later")
        self.assertEqual(len(calls), 3)

    def test_integrity_error_is_not_retried(self):
        calls = []

        def read():
            calls.append(1)
            raise ValueError("hash mismatch")

        with self.assertRaises(ValueError):
            retry_read(read)
        self.assertEqual(len(calls), 1)

    def test_nontransient_download_error_is_not_retried(self):
        calls = []

        def read():
            calls.append(1)
            raise urllib.error.HTTPError("https://example.invalid", 404, "missing", {}, None)

        with self.assertRaises(urllib.error.HTTPError):
            retry_read(read)
        self.assertEqual(len(calls), 1)


"""Offline CLI integration checks with a controlled model, never benchmark scores."""
from contextlib import ExitStack, redirect_stdout
import io
import json

import yaml


class ControlledModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, history):
        self.calls += 1
        return history + 0.125


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        torch.save({"model_state_dict": {}}, self.assets / "unet_real.pth")
        # Exercise the actual published GaussianNormalizer implementation.
        torch.save(
            (
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([4.0, 5.0, 6.0]),
                torch.tensor([2.0, 3.0, 4.0]),
                torch.tensor([3.0, 4.0, 5.0]),
            ),
            self.assets / "mean_std.pt",
        )
        (self.assets / "unet_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "dataset_name": "cylinder",
                    "model_name": "unet",
                    "N_autoregressive": 5,
                }
            )
        )
        self.velocity = np.ones((40, 64, 128, 2), dtype=np.float32)
        self.velocity[20:] = 9.0
        self.model = ControlledModel()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(cli, "prepare_assets"))
        self.stack.enter_context(patch.object(cli, "load_model", return_value=self.model))
        self.stack.enter_context(
            patch.object(cli, "load_velocity", side_effect=lambda *a: self.velocity.copy())
        )
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.args = [
            "--assets-dir",
            str(self.assets),
            "--output-dir",
            str(self.root / "output"),
            "--max-windows",
            "1",
            "--threads",
            "1",
            "--save-predictions",
        ]

    def run_cli(self, *extra):
        return cli.main(self.args + list(extra))

    def test_complete_pipeline_and_resume_without_new_prediction(self):
        result = self.run_cli()
        self.assertEqual((result["case_count"], self.model.calls), (5, 5))
        output = self.root / "output"
        for name in [
            "protocol.json",
            "cases.jsonl",
            "summary.json",
            "summary.csv",
            "report.md",
            "execution.json",
            "execution_history.jsonl",
        ]:
            self.assertTrue((output / name).is_file(), name)
        self.assertEqual(len(list((output / "predictions").glob("*.npz"))), 5)
        result = self.run_cli("--resume")
        self.assertEqual((result["reused_cached_cases"], result["new_cases"]), (5, 0))
        self.assertEqual(self.model.calls, 5)
        self.assertEqual(len((output / "execution_history.jsonl").read_text().splitlines()), 2)

    def test_future_targets_cannot_enter_model_input(self):
        captured = []
        hook = self.model.register_forward_pre_hook(
            lambda model, args: captured.append(args[0].detach().cpu().clone())
        )
        self.addCleanup(hook.remove)
        self.run_cli()
        self.assertEqual(len(captured), 5)
        # Initial pressure is physical zero; complete u/v history is physical one.
        expected = torch.tensor([0.0, -1.0 / 3.0, -0.75])
        torch.testing.assert_close(captured[0][0, 0, 0, 0], expected)
        self.assertTrue(all(x.shape == (1, 20, 64, 128, 3) for x in captured))
        self.assertTrue(all(float(x[..., :2].max()) <= 0.0 for x in captured))

    def test_resume_rejects_changed_scientific_protocol(self):
        self.run_cli()
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.run_cli("--resume", "--seed-offsets", "1")
        self.assertEqual(self.model.calls, 5)

    def test_resume_rejects_corrupted_cached_input_hash(self):
        self.run_cli()
        path = self.root / "output" / "cases.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["input_sha256"] = "corrupted"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "Cached case differs"):
            self.run_cli("--resume")
        self.assertEqual(self.model.calls, 5)

    def test_resume_requires_requested_prediction_files(self):
        self.run_cli()
        next((self.root / "output" / "predictions").glob("*.npz")).unlink()
        with self.assertRaisesRegex(ValueError, "predictions were not saved"):
            self.run_cli("--resume")
        self.assertEqual(self.model.calls, 5)


class ConfigurationAndDataTests(unittest.TestCase):
    def test_cohort_contains_all_ood_and_excludes_training(self):
        config = json.loads(cli.parse_args(["--output-dir", "unused"]).config.read_text())
        self.assertEqual({w["sim_id"] for w in config["windows"]}, set(config["official_ood_ids"]))
        self.assertEqual(len(config["windows"]), len(config["official_ood_ids"]))
        self.assertFalse(set(config["official_ood_ids"]) & set(config["released_train_sim_ids"]))
        for record in config["windows"]:
            self.assertEqual({f["column"] for f in record["range_hashes"]}, {"u", "v"})
        for asset in config["assets"].values():
            self.assertEqual(len(asset["sha256"]), 64)
            self.assertTrue(
                any(
                    rev in asset["url"]
                    for rev in [config["model_revision"], config["dataset_revision"]]
                )
            )

    def test_overlap_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = json.loads(cli.parse_args(["--output-dir", "unused"]).config.read_text())
            config["released_train_sim_ids"].append(config["official_ood_ids"][0])
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            with patch.object(cli, "load_model") as model:
                with self.assertRaisesRegex(ValueError, "overlap"):
                    cli.main(["--config", str(path), "--output-dir", str(Path(tmp) / "out")])
                model.assert_not_called()

    def test_existing_hf_loader_matches_verified_window_and_rejects_corruption(self):
        import hashlib
        from datasets import Dataset
        from realpdebench.data.fluid_hf_dataset import CylinderHFDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf = root / "cylinder/hf_dataset"
            hf.mkdir(parents=True)
            velocity = np.random.default_rng(17).normal(size=(40, 64, 128, 2)).astype(np.float32)
            fields = {
                c: np.ascontiguousarray(velocity[..., i], dtype="<f4").tobytes()
                for i, c in enumerate(["u", "v"])
            }
            Dataset.from_list(
                [dict(sim_id="1.h5", shape_t=40, shape_h=64, shape_w=128, **fields)]
            ).save_to_disk(str(hf / "real"))
            (hf / "test_index_real.json").write_text(json.dumps([dict(sim_id="1.h5", time_id=0)]))
            dataset = CylinderHFDataset(
                dataset_name="cylinder",
                dataset_root=str(root),
                dataset_type="real",
                mode="test",
                hf_auto_download=False,
            )
            record = dict(
                sim_id="1.h5",
                time_id=0,
                window_file="sample.npz",
                range_hashes=[
                    dict(column=c, sha256=hashlib.sha256(raw).hexdigest())
                    for c, raw in fields.items()
                ],
            )
            actual = cli.load_velocity(record, root / "cache", dataset=dataset)
            np.testing.assert_array_equal(actual, velocity)
            self.assertFalse((root / "cache/sample.npz").exists())
            record["range_hashes"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "differs from pinned range"):
                cli.load_velocity(record, root / "cache", dataset=dataset)
