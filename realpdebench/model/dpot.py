"""
DPOT (Discriminative Pretrained Operator Transformer) wrapper for real_benchmark.

This module provides a wrapper for the DPOT model to integrate with the benchmark project.
DPOT is a pre-trained neural operator model that handles spatio-temporal data.
"""

import torch
import os

from realpdebench.model.model import Model
from realpdebench.utils.metrics import mse_loss

# Import DPOT models
from realpdebench.model.dpot_libs.models.dpot import DPOTNet
from realpdebench.model.dpot_libs.models.dpot3d import DPOTNet3D

# Import DPOT utilities for proper resolution handling
from realpdebench.model.dpot_libs.utils.utilities import resize


class DPOT(Model):
    """
    DPOT wrapper that adapts DPOT models to the benchmark interface.
    
    Args:
        shape_in: Input shape [T, S, S, C]
        shape_out: Output shape [T, S, S, C]
        model_type: Type of DPOT model ('dpot' or 'dpot3d')
        checkpoint_path: Path to pre-trained checkpoint (optional)
        **kwargs: DPOT model parameters
    """
    
    def __init__(self, 
                 shape_in,
                 shape_out,
                 img_size=128,
                 in_channels=4,
                 out_channels=4,
                 in_timesteps=1,
                 out_timesteps=1,
                 patch_size=8,
                 embed_dim=512,
                 depth=12,
                 n_blocks=8,
                 modes=32,
                 mlp_ratio=4,
                 out_layer_dim=32,
                 normalize=False,
                 act='gelu',
                 time_agg='exp_mlp',
                 n_cls=1,
                 model_type='dpot', 
                 checkpoint_path=None, 
                 **kwargs):
        super().__init__()
        
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        
        # Extract dimensions from shapes
        # Input: [T, S, S, C] -> need [B, S, S, T, C] for DPOT
        # Store original channel count for output processing
        self.data_in_channels = shape_in[-1] # This is for data input
        self.data_out_channels = shape_out[-1] # This is for data output
        
        print(f"Data in shape: {shape_in}, data out shape: {shape_out}")
        
        self.data_in_timesteps = shape_in[0]
        self.data_out_timesteps = shape_out[0]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        
        print(f"Data input timesteps: {self.data_in_timesteps}, in_timesteps: {self.in_timesteps}")
        print(f"Data output timesteps: {self.data_out_timesteps}, out_timesteps: {self.out_timesteps}")
        
        assert self.data_in_timesteps == self.in_timesteps, f"Data input timesteps ({self.data_in_timesteps}) must be equal to in_timesteps ({self.in_timesteps})"
        assert self.data_out_timesteps >= self.out_timesteps, f"Data output timesteps ({self.data_out_timesteps}) must be greater than or equal to out_timesteps ({self.out_timesteps})"
        
        self.img_size = img_size
        
        # Default DPOT parameters - can be overridden by kwargs
        default_params = {
            'img_size': self.img_size,
            'patch_size': patch_size,
            'in_channels': in_channels,  # The config in_channels is used for model structure
            'out_channels': out_channels,  # The config out_channels is used for model structure
            'in_timesteps': in_timesteps,
            'out_timesteps': out_timesteps,  
            'embed_dim': embed_dim,
            'depth': depth,
            'n_blocks': n_blocks,
            'modes': modes,
            'mlp_ratio': mlp_ratio,
            'out_layer_dim': out_layer_dim,
            'normalize': normalize,
            'act': act,
            'time_agg': time_agg,
            'n_cls': n_cls,  # Number of datasets/classes
        }
        
        # Validate channel count
        if self.data_out_channels > 4:
            print(f"Dataset has output channels {self.data_out_channels}, but pretrained DPOT model only supports up to 4 channels. The output layer will be reinitialized.")
            assert out_channels == self.data_out_channels, f"The output channels of the model {out_channels} must be equal to the output channels of the dataset {self.data_out_channels}"

        additional_params = {
            'mixing_type': kwargs.get('mixing_type', 'afno'),
        }
        default_params.update(additional_params)
        self.model_params = default_params

        # Create the DPOT model
        if model_type == 'dpot':
            self.dpot_model = DPOTNet(**default_params)
        elif model_type == 'dpot3d':
            self.dpot_model = DPOTNet3D(**default_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
            
    def forward(self, x):
        """
        Forward pass through DPOT model in our benchmark format
        
        Args:
            x: Input tensor [B, T, S, S, C]
            
        Returns:
            Output tensor [B, T_out, S, S, C_out]
        """
        batch_size, T_in, H, W, C = x.shape
        T_out = self.data_out_timesteps
        
        assert T_in >= self.in_timesteps, f"Input timesteps ({T_in}) must be greater than or equal to in_timesteps ({self.in_timesteps})"
        assert self.out_timesteps <= T_out, f"Output timesteps {self.out_timesteps} must be less than or equal to data output timesteps {T_out}"
        
        if self.out_timesteps == T_out:
            # No sliding needed - predict all timesteps at once
            return self._forward_training_single_window(x)  # [B, T_out, S, S, C]
        
        else:
            # Sliding window 
            current_input = x
            output = []
            # Slide through the target sequence
            for t in range(0, T_out, self.out_timesteps):
                window_input = current_input[:, -self.in_timesteps:]  # [B, T_in, S, S, C]
                
                if t + self.out_timesteps > T_out:
                    # Handle the last partial window
                    remaining_steps = T_out - t
                    if remaining_steps < self.out_timesteps // 2:
                        print(f"Skipping partial window with {remaining_steps} steps")
                        # Skip if remaining steps are too few (less than half window)
                        break
                    
                    # Forward pass (still predicts out_timesteps, but we only use first remaining_steps)
                    pred = self._forward_training_single_window(window_input)  # [B, out_timesteps, S, S, C]
                    pred = pred[:, :remaining_steps]  # [B, remaining_steps, S, S, C]
                else:
                    # Forward pass
                    pred = self._forward_training_single_window(window_input)  # [B, out_timesteps, S, S, C]
                    
                    # Not the last window, update current input
                    current_input = torch.cat([current_input, pred], dim=1)
                    
                output.append(pred)
            
        output = torch.cat(output, dim=1)
        return output
    
    def _forward_training_single_window(self, x):
        """
        Native multi-step prediction for a single window
        
        Args:
            x: Input tensor [B, in_timesteps, S, S, C]
            
        Returns:
            Output tensor [B, out_timesteps, S, S, C_out]
            
        This implementation follows the exact same workflow as DPOT's evaluate_varyingres.py:
        1. Convert input format to DPOT expected format [B, S, S, T, C] 
        2. Use FFT-based resize() function to adapt to model resolution
        3. Pad channels to match pretrained model (4 channels)
        4. Process at model's native resolution
        5. Use FFT-based resize() to convert output back to original resolution
            
        """

        # Follow EXACT original DPOT workflow from evaluate_varyingres.py:
        # xx_ = resize(xx, out_size=[args.res, args.res], temporal=True)
        # im, _ = model(xx_)  
        # im = resize(im, out_size=[res, res], temporal=True)
            
        batch_size, T, H, W, C = x.shape
        original_res = [H, W]
        model_res = [self.img_size, self.img_size]
        
        # Convert format: [B, T, S, S, C] -> [B, S, S, T, C]
        x_dpot = x.permute(0, 2, 3, 1, 4)
        
        # STEP 1: Resize input to model resolution using original DPOT FFT-based resize
        if original_res != model_res:
            x_resized = resize(x_dpot, out_size=model_res, temporal=True)
        else:
            x_resized = x_dpot
        
        # STEP 2: Pad channels to 4
        B, S, S, T, C = x_resized.shape
        if C < 4:
            # we follow the original DPOT workflow to pad the channels to 4, if input channels is less than 4
            assert self.in_channels == 4, f"Even the data input channels is less than 4, the model input channels must be 4"
            x_padded = torch.ones(B, S, S, T, 4, device=x_resized.device, dtype=x_resized.dtype)
            x_padded[..., :C] = x_resized
        else:
            x_padded = x_resized
        
        # STEP 3: Native multi-step prediction
        out, _ = self.dpot_model(x_padded)  # [B, S, S, out_timesteps, C]
        
        # STEP 4: Extract original channels and resize back
        out = out[..., :self.data_out_channels]
        
        # STEP 5: Resize back to original resolution using original DPOT FFT-based resize
        if original_res != model_res:
            out = resize(out, out_size=original_res, temporal=True)
        
        # STEP 6: Convert back: [B, S, S, out_timesteps, C] -> [B, out_timesteps, S, S, C]
        output = out.permute(0, 3, 1, 2, 4)
        return output
    
    def train_loss(self, input, target):
        """
        Training loss with automatic sliding window when out_timesteps < target length
        
        Args:
            input: [B, T_in, S, S, C] - input timesteps (e.g., [B, 10, 64, 64, 1])
            target: [B, T_out, S, S, C] - target timesteps (e.g., [B, 30, 64, 64, 1])
        """
        batch_size, T_in, H, W, C = input.shape
        _, T_out, _, _, _ = target.shape
        
        assert T_in >= self.in_timesteps, f"Input timesteps ({T_in}) must be greater than or equal to in_timesteps ({self.in_timesteps})"
        assert self.out_timesteps <= T_out, f"Output timesteps {self.out_timesteps} must be less than or equal to data output timesteps {T_out}"
        
        if self.out_timesteps == T_out:
            # No sliding needed - predict all timesteps at once
            pred = self._forward_training_single_window(input)  # [B, T_out, S, S, C]
            loss = mse_loss(pred, target)
            return loss.mean()
        
        else:
            # Sliding window training
            # Combine input and target for sliding
            # full_sequence = torch.cat([input, target], dim=1)  # [B, T_in + T_out, S, S, C]
            
            total_loss = 0
            num_windows = 0
            current_input = input
            # Slide through the target sequence
            for t in range(0, T_out, self.out_timesteps):
                window_input = current_input[:, -self.in_timesteps:]  # [B, T_in, S, S, C]
                
                if t + self.out_timesteps > T_out:
                    # Handle the last partial window
                    remaining_steps = T_out - t
                    if remaining_steps < self.out_timesteps // 2:
                        print(f"Skipping partial window with {remaining_steps} steps")
                        # Skip if remaining steps are too few (less than half window)
                        break
                    
                    # For partial window, we still use the full model but only compute loss on available steps
                    window_target = target[:, t:t + remaining_steps]  # [B, remaining_steps, S, S, C]
                    
                    # Forward pass (still predicts out_timesteps, but we only use first remaining_steps)
                    pred = self._forward_training_single_window(window_input)  # [B, out_timesteps, S, S, C]
                    pred = pred[:, :remaining_steps]  # [B, remaining_steps, S, S, C]
                    
                    loss = mse_loss(pred, window_target)
                    total_loss += loss * (remaining_steps / self.out_timesteps)  # Weight by partial window size
                    num_windows += (remaining_steps / self.out_timesteps)
                else:
                    # Full window
                    window_target = target[:, t:t + self.out_timesteps]  # [B, out_timesteps, S, S, C]
                    
                    # Forward pass
                    pred = self._forward_training_single_window(window_input)  # [B, out_timesteps, S, S, C]
                    
                    loss = mse_loss(pred, window_target)
                    total_loss += loss
                    num_windows += 1
            
                    # Not the last window, update current input
                    current_input = torch.cat([current_input, pred], dim=1)
            
            if num_windows == 0:
                raise ValueError(f"No valid training windows found. out_timesteps ({self.out_timesteps}) may be too large for target length ({T_out})")
            
            return total_loss / num_windows
    
    def load_checkpoint(self, checkpoint_path, device='cpu'):
        """
        Load pre-trained checkpoint with intelligent layer matching.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            print("Warning: Loading with weights_only=False due to argparse.Namespace in checkpoint")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # If checkpoint comes from our wrapper (saved via model.state_dict()),
        # keys will be prefixed with 'dpot_model.'. Strip that prefix so they
        # match the underlying DPOT model keys.
        def strip_known_prefixes(key: str) -> str:
            prefixes = ['dpot_model.', 'module.']
            for p in prefixes:
                if key.startswith(p):
                    return key[len(p):]
            return key

        if any(k.startswith('dpot_model.') or k.startswith('module.') for k in state_dict.keys()):
            state_dict = {strip_known_prefixes(k): v for k, v in state_dict.items()}
        
        # Load checkpoint with shape matching (now that channels are compatible)
        model_state_dict = self.dpot_model.state_dict()
        compatible_state_dict = {}
        incompatible_layers = []
        
        for key, param in state_dict.items():
            if key in model_state_dict:
                if param.shape == model_state_dict[key].shape:
                    compatible_state_dict[key] = param
                else:
                    incompatible_layers.append(f"{key}: checkpoint {param.shape} vs model {model_state_dict[key].shape}")
            else:
                incompatible_layers.append(f"{key}: not found in model")
        
        # Report loading status
        total_params = len(state_dict)
        loaded_params = len(compatible_state_dict)
        
        print(f"Loading checkpoint from {checkpoint_path}")
        print(f"Compatible parameters: {loaded_params}/{total_params} ({100*loaded_params/total_params:.1f}%)")
        
        if incompatible_layers:
            print(f"Incompatible layers ({len(incompatible_layers)}):")
            for layer in incompatible_layers[:5]:  # Show first 5
                print(f"  - {layer}")
            if len(incompatible_layers) > 5:
                print(f"  ... and {len(incompatible_layers) - 5} more")
        
        # Load compatible parameters
        self.dpot_model.load_state_dict(compatible_state_dict, strict=False)
        self.dpot_model.to(device)
        
        # Report which key components were loaded
        key_components = ['blocks', 'patch_embed', 'pos_embed', 'out_layer']
        loaded_components = []
        for component in key_components:
            if any(component in key for key in compatible_state_dict.keys()):
                loaded_components.append(component)
        
        print(f"Loaded components: {', '.join(loaded_components)}")
        print(f"Checkpoint loaded successfully with {loaded_params} parameters")

        # Return metadata if present for compatibility with callers that expect it
        meta_data = None
        if isinstance(checkpoint, dict) and all(k in checkpoint for k in ['train_losses', 'val_losses', 'iteration', 'best_iteration', 'best_val_loss']):
            meta_data = {
                'all_train_losses': checkpoint['train_losses'],
                'all_val_losses': checkpoint['val_losses'],
                'iteration': checkpoint['iteration'],
                'best_iteration': checkpoint['best_iteration'],
                'best_val_loss': checkpoint['best_val_loss']
            }
        return meta_data
