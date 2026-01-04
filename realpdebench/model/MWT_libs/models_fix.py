import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import List, Tuple
import math

from realpdebench.model.MWT_libs.utils_MWT import get_filter
from realpdebench.utils.metrics import mse_loss

from functools import partial

       
class sparseKernel(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernel,self).__init__()
        
        self.k = k
        self.conv = self.convBlock(alpha*k**2, alpha*k**2)
        self.Lo = nn.Conv1d(alpha*k**2, c*k**2, 1)
        
    def forward(self, x):
        B, c, ich, Nx, Ny, T = x.shape # (B, c, ich, Nx, Ny, T)
        x = x.reshape(B, -1, Nx, Ny, T)
        x = self.conv(x)
        x = self.Lo(x.view(B, c*ich, -1)).view(B, c, ich, Nx, Ny, T)
        return x
        
        
    def convBlock(self, ich, och):
        net = nn.Sequential(
            nn.Conv3d(och, och, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        return net 


def compl_mul3d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)
    

    
# fft conv taken from: https://github.com/zongyi-li/fourier_neural_operator
class sparseKernelFT(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernelFT, self).__init__()        
        
        self.modes = alpha

        self.weights1 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, 2))
        self.weights2 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, 2))        
        self.weights3 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, 2))        
        self.weights4 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, 2))        
        nn.init.xavier_normal_(self.weights1)
        nn.init.xavier_normal_(self.weights2)
        nn.init.xavier_normal_(self.weights3)
        nn.init.xavier_normal_(self.weights4)
        
        self.Lo = nn.Conv1d(c*k**2, c*k**2, 1)
#         self.Wo = nn.Conv1d(c*k**2, c*k**2, 1)
        self.k = k
        
    def forward(self, x):
        B, c, ich, Nx, Ny, T = x.shape # (B, c, ich, N, N, T)
        
        x = x.reshape(B, -1, Nx, Ny, T)
        # x_fft = torch.rfft(x, 3, normalized=True, onesided=True)
        x_fft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
        
        # Multiply relevant Fourier modes
        l1 = min(self.modes, Nx//2+1)
        l2 = min(self.modes, Ny//2+1)
        out_ft = torch.zeros(B, c*ich, Nx, Ny, T//2 +1, 2, device=x.device)
        
        out_ft[:, :, :l1, :l2, :self.modes] = compl_mul3d(
            x_fft[:, :, :l1, :l2, :self.modes], self.weights1[:, :, :l1, :l2, :])
        out_ft[:, :, -l1:, :l2, :self.modes] = compl_mul3d(
                x_fft[:, :, -l1:, :l2, :self.modes], self.weights2[:, :, :l1, :l2, :])
        out_ft[:, :, :l1, -l2:, :self.modes] = compl_mul3d(
                x_fft[:, :, :l1, -l2:, :self.modes], self.weights3[:, :, :l1, :l2, :])
        out_ft[:, :, -l1:, -l2:, :self.modes] = compl_mul3d(
                x_fft[:, :, -l1:, -l2:, :self.modes], self.weights4[:, :, :l1, :l2, :])
        
        #Return to physical space
        # x = torch.irfft(out_ft, 3, normalized=True, onesided=True, signal_sizes=(Nx, Ny, T))
        x = torch.fft.irfftn(out_ft, s=(Nx, Ny, T), dim=(-3, -2, -1), norm="ortho")

        x = F.relu(x)
        x = self.Lo(x.view(B, c*ich, -1)).view(B, c, ich, Nx, Ny, T)
        return x
        
    
class MWT_CZ(nn.Module):
    def __init__(self,
                 k = 3, alpha = 5, 
                 L = 0, c = 1,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT_CZ, self).__init__()
        
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0@PHI0
        G0r = G0@PHI0
        H1r = H1@PHI1
        G1r = G1@PHI1
        
        H0r[np.abs(H0r)<1e-8]=0
        H1r[np.abs(H1r)<1e-8]=0
        G0r[np.abs(G0r)<1e-8]=0
        G1r[np.abs(G1r)<1e-8]=0
        
        self.A = sparseKernelFT(k, alpha, c)
        self.B = sparseKernelFT(k, alpha, c)
        self.C = sparseKernelFT(k, alpha, c)
        
        self.T0 = nn.Conv1d(c*k**2, c*k**2, 1)

        if initializer is not None:
            self.reset_parameters(initializer)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((np.kron(H0, H0).T, 
                            np.kron(H0, H1).T,
                            np.kron(H1, H0).T,
                            np.kron(H1, H1).T,
                           ), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((np.kron(G0, G0).T,
                            np.kron(G0, G1).T,
                            np.kron(G1, G0).T,
                            np.kron(G1, G1).T,
                           ), axis=0)))
        
        self.register_buffer('rc_ee', torch.Tensor(
            np.concatenate((np.kron(H0r, H0r), 
                            np.kron(G0r, G0r),
                           ), axis=0)))
        self.register_buffer('rc_eo', torch.Tensor(
            np.concatenate((np.kron(H0r, H1r), 
                            np.kron(G0r, G1r),
                           ), axis=0)))
        self.register_buffer('rc_oe', torch.Tensor(
            np.concatenate((np.kron(H1r, H0r), 
                            np.kron(G1r, G0r),
                           ), axis=0)))
        self.register_buffer('rc_oo', torch.Tensor(
            np.concatenate((np.kron(H1r, H1r), 
                            np.kron(G1r, G1r),
                           ), axis=0)))
        
        
    def forward(self, x):
        
        B, c, ich, Nx, Ny, T = x.shape # (B, c, k^2, Nx, Ny, T)
        ns = math.floor(np.log2(Nx))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

#         decompose
        for i in range(ns-self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x.reshape(B, c*ich, -1)).view(
            B, c, ich, 2**self.L, 2**self.L, T) # coarsest scale transform

#        reconstruct            
        for i in range(ns-1-self.L,-1,-1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), 2)
            x = self.evenOdd(x)

        return x

    
    def wavelet_transform(self, x):
        xa = torch.cat([x[:, :, :, ::2 , ::2 , :], 
                        x[:, :, :, ::2 , 1::2, :], 
                        x[:, :, :, 1::2, ::2 , :], 
                        x[:, :, :, 1::2, 1::2, :]
                       ], 2)
        waveFil = partial(torch.einsum, 'bcixyt,io->bcoxyt') 
        d = waveFil(xa, self.ec_d)
        s = waveFil(xa, self.ec_s)
        return d, s
        
        
    def evenOdd(self, x):
        
        B, c, ich, Nx, Ny, T = x.shape # (B, c, 2*k^2, Nx, Ny)
        assert ich == 2*self.k**2
        evOd = partial(torch.einsum, 'bcixyt,io->bcoxyt')
        x_ee = evOd(x, self.rc_ee)
        x_eo = evOd(x, self.rc_eo)
        x_oe = evOd(x, self.rc_oe)
        x_oo = evOd(x, self.rc_oo)
        
        x = torch.zeros(B, c, self.k**2, Nx*2, Ny*2, T,
            device = x.device)
        x[:, :, :, ::2 , ::2 , :] = x_ee
        x[:, :, :, ::2 , 1::2, :] = x_eo
        x[:, :, :, 1::2, ::2 , :] = x_oe
        x[:, :, :, 1::2, 1::2, :] = x_oo
        return x
    
    def reset_parameters(self, initializer):
        initializer(self.T0.weight)
    
    
class MWT3d(nn.Module):
    def __init__(self,
                 ich = 1, k = 3, alpha = 2, c = 1,
                 nCZ = 3,
                 L = 0,
                 base = 'legendre',
                 initializer = None,
                 n_targets = None,
                 **kwargs):
        super(MWT3d,self).__init__()
        
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk = nn.Linear(ich, c*k**2)
        self.n_targets = n_targets

        self.MWT_CZ = nn.ModuleList(
            [MWT_CZ(k, alpha, L, c, base, 
            initializer) for _ in range(nCZ)]
        )
        self.BN = nn.ModuleList(
            [nn.BatchNorm3d(c*k**2) for _ in range(nCZ)]
        )
        self.Lc0 = nn.Linear(c*k**2, 128)
        self.Lc1 = nn.Linear(128, n_targets)
        
        if initializer is not None:
            self.reset_parameters(initializer)
        
    def forward(self, x):
        pdb.set_trace()
        switch = False
        if x.dim() == 5 and x.shape[-1] < x.shape[1]:
            switch = True
            x = x.permute(0, 2, 3, 1, 4)  # (B, T, H, W, C) -> (B, H, W, T, C)

        B, Nx, Ny, T, ich = x.shape # (B, Nx, Ny, T, d)
        ns = math.floor(np.log2(Nx))
        x = self.Lk(x)
        x = x.view(B, Nx, Ny, T, self.c, self.k**2)
        x = x.permute(0, 4, 5, 1, 2, 3)
    
        for i in range(self.nCZ):
            x = self.MWT_CZ[i](x)
            x = self.BN[i](x.view(B, -1, Nx, Ny, T)).view(
                B, self.c, self.k**2, Nx, Ny, T)
            if i < self.nCZ-1:
                x = F.relu(x)

        x = x.view(B, -1, Nx, Ny, T) # collapse c and k**2
        x = x.permute(0, 2, 3, 4, 1)
        x = self.Lc0(x)
        x = F.relu(x)
        x = self.Lc1(x).squeeze()
        if switch:
            x = x.permute(0, 3, 1, 2, 4)
        return x

    def train_loss(self, input, target):
        pred = self.forward(input)
        
        return mse_loss(pred, target)


    def reset_parameters(self, initializer):
        initializer(self.Lc0.weight)
        initializer(self.Lc1.weight)