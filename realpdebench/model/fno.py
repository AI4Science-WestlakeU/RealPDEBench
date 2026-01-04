"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed 
in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf), which takes the 2D spatial + 1D
temporal equation directly as a 3D problem
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from realpdebench.model.model import Model
from realpdebench.utils.metrics import mse_loss

# 3d fourier layers
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, \
                                    self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, \
                                    self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, \
                                    self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, \
                                    self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, \
                            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(Model):
    def __init__(self, modes1, modes2, modes3, n_layers, width, shape_in, shape_out):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains n layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. n layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width

        self.shape_in = shape_in
        self.shape_out = shape_out
        self.dim_in = shape_in[-1]
        self.dim_out = shape_out[-1] * shape_out[0] // shape_in[0] # C_out * T_out / T_in
        self.padding = 6 # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(self.dim_in+3, self.width)
        # the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.n_layers = n_layers
        self.spectral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            self.spectral_convs.append(
                SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3))
            self.convs.append(nn.Conv3d(self.width, self.width, 1))
            self.bns.append(nn.BatchNorm3d(self.width))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.dim_out)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device) # [B,T,S,S,C_in]
        x = torch.cat((x, grid), dim=-1) # [B,T,S,S,C_in+3]
        x = self.fc0(x) # [B,T,S,S,width]
        x = x.permute(0, 4, 1, 2, 3) # [B,width,T,S,S]
        # pad the domain if input is non-periodic
        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding]) 

        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.convs[i](x)
            x = x1 + x2
            x = self.bns[i](x)
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = x[..., :-self.padding, :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) 
        x = self.fc1(x) # [B,T,S,S,128]
        x = F.gelu(x)
        x = self.fc2(x) # [B,T,S,S,C_out]

        x = x.reshape(*x.shape[:-1], self.shape_out[-1], self.shape_out[0] // self.shape_in[0]) 
        out = x.permute(0, 1, 5, 2, 3, 4).reshape(x.shape[0], *self.shape_out) 
        return out

    def train_loss(self, input, target):
        pred = self.forward(input)
        return mse_loss(pred, target)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)