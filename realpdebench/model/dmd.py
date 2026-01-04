import numpy as np
from scipy.linalg import svd
from typing import Optional, Tuple
import warnings
import torch
from torch import nn

class DMD(nn.Module):
    def __init__(self, n_modes, n_predict, input_feature, n_autoregressive, rank: Optional[int] = None):
        super().__init__()
        self.n_modes = n_modes
        self.n_predict = n_predict
        self.rank = rank
        self.input_feature = input_feature
        self.N_autoregressive = n_autoregressive
        self.modes = None  
        self.eigenvalues = None  
        self.amplitudes = None  
        self.original_shape = None
        self.train_time_steps = None  
        
    def fit(self, flow_field: np.ndarray) -> 'DMD':
        time_steps, height, width, components = flow_field.shape
        
        # Save original shape and training time steps
        self.original_shape = (height, width, components)
        self.train_time_steps = time_steps  
        
        # Reshape to snapshot matrix: (time, space*components)
        n_space = height * width * components
        snapshot_matrix = flow_field.reshape(time_steps, n_space).T  # (n_space, n_snapshots)
        
        # Build data matrix pair: X1 and X2
        # X1: first n-1 snapshots, X2: last n-1 snapshots
        X1 = snapshot_matrix[:, :-1]  # (n_space, n_snapshots-1)
        X2 = snapshot_matrix[:, 1:]   # (n_space, n_snapshots-1)
        
        # Perform SVD decomposition on X1
        U, s, Vt = svd(X1, full_matrices=False)
        
        # Determine truncation rank
        if self.rank is not None:
            rank = min(self.rank, len(s))
        else:
            rank = len(s)
        
        # Truncate SVD
        U = U[:, :rank]  # (n_space, rank)
        s = s[:rank]  # (rank,)
        Vt = Vt[:rank, :]  # (rank, n_snapshots-1)
        
        # Compute low-dimensional evolution operator: A_tilde = U^T @ X2 @ V @ S^{-1}
        S_inv = np.diag(1.0 / s)
        V = Vt.T  # (n_snapshots-1, rank)
        A_tilde = U.T @ X2 @ V @ S_inv  # (rank, rank)
        
        # Perform eigenvalue decomposition on A_tilde
        eigenvalues, W = np.linalg.eig(A_tilde)
        
        # Compute DMD modes: Psi = X2 @ V @ S^{-1} @ W
        self.modes = X2 @ V @ S_inv @ W  # (n_space, rank)
        self.eigenvalues = eigenvalues  # (rank,)
        
        # Compute DMD amplitudes (for initial condition)
        initial_condition = snapshot_matrix[:, 0]  # (n_space,)
        b = np.linalg.lstsq(self.modes, initial_condition, rcond=None)[0]
        self.amplitudes = b  # (rank,)
        
        # If number of modes is specified, keep only the first n_modes
        if self.n_modes is not None and self.n_modes < len(self.eigenvalues):
            # Sort by amplitude magnitude
            idx = np.argsort(np.abs(self.amplitudes))[::-1]
            idx = idx[:self.n_modes]
            self.modes = self.modes[:, idx]
            self.eigenvalues = self.eigenvalues[idx]
            self.amplitudes = self.amplitudes[idx]
        
        return self
    
    def predict(self, n_steps: int = 10, dt: float = 1.0) -> np.ndarray:
        """
        Predict future n_steps time steps using DMD
        
        DMD prediction formula: x(t) = Σ b_i * ψ_i * exp(λ_i * t)
        
        Args:
            n_steps: Number of time steps to predict, default 10
            dt: Time step size, default 1.0
        
        Returns:
            Predicted flow field, shape (n_steps, height, width, components)
        """
        if self.modes is None:
            raise ValueError("Please call fit method first to perform DMD decomposition")
        
        n_modes = self.modes.shape[1]
        n_space = self.modes.shape[0]
        
        # Time points: start prediction from after the last time step of training data
        # 
        # Time index explanation:
        # - Training data: t = 0, 1, 2, ..., train_time_steps-1
        #   Example: 20 time steps → t = 0, 1, 2, ..., 19
        # - Prediction data: t = train_time_steps, train_time_steps+1, ..., train_time_steps+n_steps-1
        #   Example: predict 20 time steps → t = 20, 21, 22, ..., 39
        #
        # This ensures prediction is continuous, starting from after the last time step of training data
        start_time = self.train_time_steps  # Start from after the last time step of training data
        t = np.arange(start_time, start_time + n_steps) * dt
        
        # Predict each time step
        predicted_snapshots = []
        for ti in t:
            # DMD prediction: x(t) = Σ b_i * ψ_i * exp(λ_i * t)
            prediction = np.zeros(n_space, dtype=complex)
            for i in range(n_modes):
                prediction += self.amplitudes[i] * self.modes[:, i] * np.exp(self.eigenvalues[i] * ti * 0.0025)
            
            predicted_snapshots.append(prediction)
        
        # Convert to array and take real part
        predicted_snapshots = np.array(predicted_snapshots).real  # (n_steps, n_space)
        
        # Reshape back to original shape
        height, width, components = self.original_shape
        predicted_snapshots = predicted_snapshots.reshape(n_steps, height, width, components)
        
        return predicted_snapshots


    def dmd_predict(
        self,
        input_frames: np.ndarray,
        n_modes: int = 5,
        n_predict: int = 10
    ) -> np.ndarray:
        """
        Convenience function: Use DMD to predict future frames based on input frames
        
        Args:
            input_frames: Input frames, shape (time, 64, 128, 2)
            n_modes: Number of DMD modes to retain, default 5
            n_predict: Number of time steps to predict, default 10
        
        Returns:
            Predicted future frames, shape (n_predict, 64, 128, 2)
        
        Example:
            >>> input_frames = np.random.randn(10, 64, 128, 2)  # First 10 frames
            >>> predicted = dmd_predict(input_frames, n_modes=5, n_predict=10)
            >>> print(predicted.shape)  # (10, 64, 128, 2)
        """
        if input_frames.ndim != 4:
            raise ValueError(f"input_frames should be a 4D array (time, height, width, components), but got {input_frames.ndim}D")
        
        if input_frames.shape[0] < 2:
            raise ValueError("At least 2 frames are required for DMD decomposition")
        
        # Create DMD model and train
        dmd = DMD(n_modes=n_modes, n_predict=n_predict, input_feature=input_frames.shape[-1], n_autoregressive=self.N_autoregressive)
        dmd.fit(input_frames)
        
        # Predict future n_predict frames
        predicted = dmd.predict(n_steps=n_predict)
        
        return predicted


    def dmd_predict_batch(
        self,
        input_batch: np.ndarray,
        n_modes: int = 5,
        n_predict: int = 10
    ) -> np.ndarray:
        """
        Batch prediction: Perform DMD prediction on multiple samples
        
        Args:
            input_batch: Input batch, shape (batch_size, time, 64, 128, 2)
            n_modes: Number of DMD modes to retain, default 5
            n_predict: Number of time steps to predict, default 10
        
        Returns:
            Prediction results, shape (batch_size, n_predict, 64, 128, 2)
        
        Example:
            >>> input_batch = np.random.randn(5, 10, 64, 128, 2)  # 5 samples, each with 10 frames
            >>> predicted = dmd_predict_batch(input_batch, n_modes=5, n_predict=10)
            >>> print(predicted.shape)  # (5, 10, 64, 128, 2)
        """
        if input_batch.ndim != 5:
            raise ValueError(f"input_batch should be a 5D array (batch_size, time, height, width, components), but got {input_batch.ndim}D")
        
        batch_size = input_batch.shape[0]
        predictions = []
        
        for i in range(batch_size):
            input_frames = input_batch[i]  # (10, 64, 128, 2)
            predicted = self.dmd_predict(input_frames, n_modes=n_modes, n_predict=n_predict)
            predictions.append(predicted)
        
        return np.array(predictions)

    def forward(self, input):
        input_np = input.cpu().numpy()  # (batch_size, 20, 64, 128, 3)
        input_frames_20 = input_np[:, :, :, :, :self.input_feature]  # (batch_size, 10, 64, 128, 3)
        pred_np = self.dmd_predict_batch(input_frames_20, n_modes=self.n_modes, n_predict=self.n_predict)  # (batch_size, 20, 64, 128, 3)
        pred = torch.from_numpy(pred_np).to(input.device).float()
        return pred

    def eval(self):
        pass

    def parameters(self, recurse=True):
        return iter(())

    def load_checkpoint(self, checkpoint_path, device):
        return checkpoint_path
    
