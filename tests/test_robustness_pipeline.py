"""Offline CLI integration checks with a controlled model, never benchmark scores."""
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'robustness_pipeline_cli', ROOT / 'scripts/eval_missing_observations.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


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
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        torch.save({'model_state_dict': {}}, self.assets / 'unet_real.pth')
        # Exercise the actual published GaussianNormalizer implementation.
        torch.save((torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.]),
                    torch.tensor([2., 3., 4.]), torch.tensor([3., 4., 5.])),
                   self.assets / 'mean_std.pt')
        (self.assets / 'unet_config.yaml').write_text(yaml.safe_dump({
            'dataset_name': 'cylinder', 'model_name': 'unet',
            'N_autoregressive': 5}))
        self.velocity = np.ones((40, 64, 128, 2), dtype=np.float32)
        self.velocity[20:] = 9.0
        self.model = ControlledModel()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(cli, 'prepare_assets'))
        self.stack.enter_context(patch.object(cli, 'load_model', return_value=self.model))
        self.stack.enter_context(patch.object(cli, 'load_velocity', side_effect=lambda *a: self.velocity.copy()))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.args = ['--assets-dir', str(self.assets), '--output-dir',
                     str(self.root / 'output'), '--max-windows', '1',
                     '--threads', '1', '--save-predictions']

    def run_cli(self, *extra):
        return cli.main(self.args + list(extra))

    def test_complete_pipeline_and_resume_without_new_prediction(self):
        result = self.run_cli()
        self.assertEqual((result['case_count'], self.model.calls), (5, 5))
        output = self.root / 'output'
        for name in ['protocol.json', 'cases.jsonl', 'summary.json', 'summary.csv',
                     'report.md', 'execution.json', 'execution_history.jsonl']:
            self.assertTrue((output / name).is_file(), name)
        self.assertEqual(len(list((output / 'predictions').glob('*.npz'))), 5)
        result = self.run_cli('--resume')
        self.assertEqual((result['reused_cached_cases'], result['new_cases']), (5, 0))
        self.assertEqual(self.model.calls, 5)
        self.assertEqual(len((output / 'execution_history.jsonl').read_text().splitlines()), 2)

    def test_future_targets_cannot_enter_model_input(self):
        captured = []
        hook = self.model.register_forward_pre_hook(
            lambda model, args: captured.append(args[0].detach().cpu().clone()))
        self.addCleanup(hook.remove)
        self.run_cli()
        self.assertEqual(len(captured), 5)
        # Initial pressure is physical zero; complete u/v history is physical one.
        expected = torch.tensor([0., -1. / 3., -0.75])
        torch.testing.assert_close(captured[0][0, 0, 0, 0], expected)
        self.assertTrue(all(x.shape == (1, 20, 64, 128, 3) for x in captured))
        self.assertTrue(all(float(x[..., :2].max()) <= 0.0 for x in captured))

    def test_resume_rejects_changed_scientific_protocol(self):
        self.run_cli()
        with self.assertRaisesRegex(ValueError, 'protocol'):
            self.run_cli('--resume', '--seed-offsets', '1')
        self.assertEqual(self.model.calls, 5)

    def test_resume_rejects_corrupted_cached_input_hash(self):
        self.run_cli()
        path = self.root / 'output' / 'cases.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['input_sha256'] = 'corrupted'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'Cached case differs'):
            self.run_cli('--resume')
        self.assertEqual(self.model.calls, 5)

    def test_resume_requires_requested_prediction_files(self):
        self.run_cli()
        next((self.root / 'output' / 'predictions').glob('*.npz')).unlink()
        with self.assertRaisesRegex(ValueError, 'predictions were not saved'):
            self.run_cli('--resume')
        self.assertEqual(self.model.calls, 5)


if __name__ == '__main__':
    unittest.main()
