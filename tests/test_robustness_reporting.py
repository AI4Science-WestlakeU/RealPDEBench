import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import numpy as np
from realpdebench.robustness.metrics import score_velocity
from realpdebench.robustness.io import retry_read

path=Path(__file__).resolve().parents[1]/'scripts/eval_missing_observations.py'
spec=importlib.util.spec_from_file_location('robustness_cli',path)
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


class ReportingTests(unittest.TestCase):
    def test_metrics_exact_forecast_and_empty_occlusion(self):
        x=np.ones((20,4,5,2),np.float32);fluid=np.ones((4,5),bool)
        m=score_velocity(x,x,fluid,np.zeros_like(fluid))
        self.assertEqual(m['all_fluid']['relative_l2'],0)
        self.assertIsNone(m['occluded'])
        self.assertEqual(m['fluctuation_ke_mae_full_grid'],0)

    def test_metric_spatial_regions_are_scored_separately(self):
        target=np.ones((20,4,5,2),np.float32);pred=target.copy();pred[:,0,0]=0
        missing=np.zeros((4,5),bool);missing[0,0]=True
        m=score_velocity(pred,target,np.ones_like(missing),missing)
        self.assertEqual(m['observed']['relative_l2'],0)
        self.assertEqual(m['occluded']['relative_l2'],1)

    def test_report_weights_trajectories_not_number_of_seeds(self):
        rows=[]
        for sim,error in [('a',1.),('a',3.),('b',8.)]:
            metric={'all_fluid':{'relative_l2':error}}
            rows.append({'case':{'sim_id':sim,'missing_fraction':.3,'mask_kind':'scattered','fill':'zero'},'unet':metric,'persistence':metric})
        protocol={'cohort':'controlled test','blocks':1,'device':'cpu','selected_windows':2,'available_windows':2}
        with tempfile.TemporaryDirectory() as tmp:
            result=cli.report(Path(tmp),rows,protocol)
            for item in result['summary']:self.assertEqual(item['mean_fluid_relative_l2'],5.)
            for name in ['summary.json','summary.csv','report.md']:self.assertTrue((Path(tmp)/name).exists())

    def test_primary_default_is_explicit_one_block(self):
        args=cli.parse_args(['--output-dir','unused'])
        self.assertEqual(args.blocks,1)
        self.assertEqual(args.device,'cpu')
        self.assertEqual(args.seed_offsets,[0])

    def test_existing_outputs_require_explicit_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'existing.txt').write_text('not a fresh run')
            with self.assertRaises(FileExistsError):cli.main(['--output-dir',tmp])

    def test_transient_download_error_is_retried(self):
        calls=[]
        def read():
            calls.append(1)
            if len(calls)<3:raise urllib.error.HTTPError('https://example.invalid',503,'temporary',{},None)
            return b'verified later'
        with patch('realpdebench.robustness.io.time.sleep'):
            self.assertEqual(retry_read(read),b'verified later')
        self.assertEqual(len(calls),3)

    def test_integrity_error_is_not_retried(self):
        calls=[]
        def read():calls.append(1);raise ValueError('hash mismatch')
        with self.assertRaises(ValueError):retry_read(read)
        self.assertEqual(len(calls),1)

    def test_nontransient_download_error_is_not_retried(self):
        calls=[]
        def read():calls.append(1);raise urllib.error.HTTPError('https://example.invalid',404,'missing',{},None)
        with self.assertRaises(urllib.error.HTTPError):retry_read(read)
        self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
