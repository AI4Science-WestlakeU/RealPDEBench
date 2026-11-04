"""Small CPU contract tests; no external datasets or checkpoints required."""
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from realpdebench.model.load_model import load_model
from realpdebench.model.transolver_sigma import TransolverSigma


class TransolverSigmaTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        self.options = dict(n_hidden=16, n_head=2, n_layers=2, slice_num=4,
                            spectral_modes_d=8, spectral_modes_hw=16, ref=2)

    def test_channel_layouts_and_backward(self):
        # Includes temporal/spatial FFT mode clipping and unequal channel counts.
        for channels_in, channels_out in ((5, 3), (3, 3), (16, 16)):
            with self.subTest(channels=(channels_in, channels_out)):
                shape_in = (10, 6, 8, channels_in)
                shape_out = (10, 6, 8, channels_out)
                model = TransolverSigma(shape_in, shape_out, **self.options)
                inputs = torch.randn(2, *shape_in)
                targets = torch.randn(2, *shape_out)
                prediction = model(inputs)
                self.assertEqual(prediction.shape, targets.shape)
                loss = model.train_loss(inputs, targets).mean()
                torch.testing.assert_close(loss, (prediction - targets).square().mean())
                loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_factory_and_checkpoint_round_trip(self):
        shape = (4, 6, 8, 3)
        inputs, targets = torch.randn(*shape), torch.randn(*shape)
        options = dict(self.options, model_name='transolver_sigma', mlp_ratio=2,
                       dropout=0., act='gelu')
        model = load_model([(inputs, targets)], **options).eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pth'
            torch.save(dict(model_state_dict=model.state_dict(), train_losses=[],
                            val_losses={}, iteration=1, best_iteration=1,
                            best_val_loss=0.), checkpoint)
            restored = load_model([(inputs, targets)], **options).eval()
            metadata = restored.load_checkpoint(checkpoint, 'cpu')
            self.assertEqual(metadata['iteration'], 1)
            torch.testing.assert_close(model(inputs[None]), restored(inputs[None]))

    def test_incompatible_shapes_fail_explicitly(self):
        shape = (4, 6, 8, 3)
        with self.assertRaisesRegex(ValueError, 'must match'):
            TransolverSigma(shape, (8, 6, 8, 3), **self.options)
        with self.assertRaisesRegex(ValueError, 'divisible'):
            TransolverSigma(shape, shape, n_hidden=18, n_head=2)
        model = TransolverSigma(shape, shape, **self.options)
        with self.assertRaisesRegex(ValueError, 'Expected'):
            model(torch.randn(1, 4, 8, 6, 3))

    def test_supplied_configurations_and_parameter_counts(self):
        scenarios = {
            'cylinder': ((20, 64, 128, 3), (20, 64, 128, 3), 3118628),
            'controlled_cylinder': ((10, 64, 128, 5), (10, 64, 128, 3), 3115270),
            'fsi': ((20, 64, 64, 3), (20, 64, 64, 3), 3118628),
            'foil': ((20, 64, 128, 3), (20, 64, 128, 3), 3118628),
            'combustion': ((20, 64, 64, 16), (20, 64, 64, 16), 3155496),
        }
        config_root = Path(__file__).resolve().parents[1] / 'realpdebench/configs'
        for scenario, (shape_in, shape_out, parameter_count) in scenarios.items():
            with self.subTest(scenario=scenario):
                config = yaml.safe_load(
                    (config_root / scenario / 'transolver_sigma.yaml').read_text())
                self.assertEqual(config['dataset_name'], scenario)
                self.assertEqual(config['num_update'], 20000)
                self.assertEqual(config['train_batch_size'], 4)
                self.assertEqual(config['scheduler'], 'cosine')
                self.assertEqual(config['lr'], 0.0005 if scenario == 'combustion' else 0.001)
                self.assertEqual(config['checkpoint_path'], '')
                model = load_model([(torch.zeros(shape_in), torch.zeros(shape_out))], **config)
                self.assertEqual(sum(p.numel() for p in model.parameters()), parameter_count)
                self.assertEqual((len(model.stage1), len(model.stage2)), (4, 4))


if __name__ == '__main__':
    unittest.main()
