import unittest
import numpy as np
import torch
from realpdebench.robustness.missing_observation import make_missing_mask, fill_history, autoregressive_predict

class Tests(unittest.TestCase):
 def setUp(self):self.fluid=np.ones((12,16),bool);self.history=np.ones((3,12,16,3),np.float32);self.history[...,2]=7
 def test_exact_matching_counts(self):
  for kind in ['scattered','compact_patch']:
   m=make_missing_mask(self.fluid,.3,kind,7);self.assertEqual(m.sum(),round(.3*self.fluid.sum()))
 def test_constant_nearest_and_pressure_preserved(self):
  m=make_missing_mask(self.fluid,.3,'compact_patch',7);a=fill_history(self.history,self.fluid,m,'nearest_observed');self.assertTrue(np.array_equal(a,self.history))
 def test_zero_observed_unchanged_and_input_unmodified(self):
  m=make_missing_mask(self.fluid,.3,'scattered',7);a=fill_history(self.history,self.fluid,m,'zero');self.assertTrue(np.array_equal(a[:,~m],self.history[:,~m]));self.assertTrue(np.all(self.history[...,:2]==1));self.assertTrue(np.all(a[...,2]==7))
 def test_no_missing_copy(self):
  a=fill_history(self.history,self.fluid,np.zeros_like(self.fluid),'zero');self.assertTrue(np.array_equal(a,self.history));self.assertIsNot(a,self.history)
 def test_bad_fraction(self):
  with self.assertRaises(ValueError):make_missing_mask(self.fluid,1.,'scattered',0)
 def test_unknown_fill(self):
  with self.assertRaises(ValueError):fill_history(self.history,self.fluid,np.zeros_like(self.fluid),'invented')
 def test_obstacle_missing_rejected(self):
  self.fluid[0,0]=False;m=np.zeros_like(self.fluid);m[0,0]=True
  with self.assertRaises(ValueError):fill_history(self.history,self.fluid,m,'zero')
 def test_two_block_normalization_and_all_channels(self):
  class Norm:
   mi=torch.tensor([1.,2.,3.]);si=torch.tensor([2.,4.,8.]);mt=torch.tensor([5.,6.,7.]);st=torch.tensor([3.,6.,9.])
   def preprocess(self,x,y):return (x-self.mi)/self.si,(y-self.mt)/self.st
   def postprocess(self,x,y):return x*self.si+self.mi,y*self.st+self.mt
  class Model(torch.nn.Module):
   def forward(self,x):return x+.25
  model=Model().eval();a=autoregressive_predict(model,Norm(),np.zeros((1,2,2,3),np.float32),blocks=2)
  expected=torch.tensor([[4.25,4.5,5.875],[10.625,11.25,12.484375]])
  self.assertTrue(torch.equal(a[0,:,0,0],expected))
 def test_training_mode_rejected(self):
  with self.assertRaises(ValueError):autoregressive_predict(torch.nn.Identity(),None,self.history)
 def test_invalid_block_count(self):
  with self.assertRaises(ValueError):autoregressive_predict(torch.nn.Identity().eval(),None,self.history,blocks=0)
 def test_cpu_tensor_history_and_device(self):
  class Norm:
   device='cpu'
   def preprocess(self,x,y):return x.to(self.device),y.to(self.device)
   def postprocess(self,x,y):return x.to(self.device),y.to(self.device)
  model=torch.nn.Identity().eval()
  actual=autoregressive_predict(model,Norm(),torch.from_numpy(self.history),blocks=2,device='cpu')
  self.assertEqual(actual.device.type,'cpu')
  self.assertTrue(torch.equal(actual,torch.from_numpy(np.concatenate([self.history,self.history]))[None]))
 def test_normalizer_device_mismatch_rejected(self):
  class Norm:device='meta'
  with self.assertRaisesRegex(ValueError,'normalizer'):
   autoregressive_predict(torch.nn.Identity().eval(),Norm(),self.history,device='cpu')
 def test_unknown_device_rejected(self):
  with self.assertRaises(ValueError):autoregressive_predict(torch.nn.Identity().eval(),None,self.history,device='meta')
 def test_boolean_blocks_rejected(self):
  with self.assertRaises(ValueError):autoregressive_predict(torch.nn.Identity().eval(),None,self.history,blocks=True)
 def test_nonfinite_input_rejected(self):
  x=self.history.copy();x[0,0,0,0]=np.nan
  with self.assertRaises(ValueError):autoregressive_predict(torch.nn.Identity().eval(),None,x)
 def test_buffer_only_model_device_mismatch_rejected_before_forward(self):
  class Model(torch.nn.Module):
   def __init__(self):super().__init__();self.register_buffer('offset',torch.empty((),device='meta'))
   def forward(self,x):raise AssertionError('Device mismatch must fail before forward')
  with self.assertRaisesRegex(ValueError,'Move the model'):
   autoregressive_predict(Model().eval(),None,self.history,device='cpu')
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware unavailable')
 def test_cuda_prediction_retains_device_and_values(self):
  class Norm:
   device=torch.device('cuda',torch.cuda.current_device())
   def preprocess(self,x,y):return x.to(self.device),y.to(self.device)
   def postprocess(self,x,y):return x.to(self.device),y.to(self.device)
  class Model(torch.nn.Module):
   def __init__(self):super().__init__();self.register_buffer('offset',torch.tensor(.25))
   def forward(self,x):return x+self.offset
  model=Model().cuda().eval()
  output=autoregressive_predict(model,Norm(),self.history,blocks=2,device='cuda')
  expected=np.concatenate([self.history+.25,self.history+.5])[None]
  self.assertEqual(output.device.type,'cuda')
  np.testing.assert_allclose(output.cpu().numpy(),expected)


if __name__ == "__main__":
    unittest.main()
