import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import List, Tuple
import math

from realpdebench.model.MWT_libs.utils_MWT import get_filter
from realpdebench.utils.metrics import mse_loss


class sparseKernel1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernel1d,self).__init__()
       
        self.k = k
        self.Li = nn.Linear(c*k, 128)
        self.conv = self.convBlock(c*k, 128)
        self.Lo = nn.Linear(128, c*k)
       
    def forward(self, x):
        B, N, c, ich = x.shape # (B, N, c, k)
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.Lo(x)
        x = x.view(B, N, c, ich)
        return x
       
       
    def convBlock(self, ich, och):
        net = nn.Sequential(
            nn.Conv1d(ich, och, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        return net

def compl_mul1d(x, weights):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", x, weights)

class sparseKernelFT1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernelFT1d, self).__init__()       
       
        self.modes1 = alpha
        self.scale = (1 / (c*k*c*k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c*k, c*k, self.modes1, dtype=torch.cfloat))
        self.weights1.requires_grad = True
        self.k = k
       
    def forward(self, x):
        B, N, c, k = x.shape # (B, N, c, k)
       
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N//2+1)
        out_ft = torch.zeros(B, c*k, N//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x


class MWT_CZ1d(nn.Module):
    def __init__(self,
                 k = 3, alpha = 5,
                 L = 0, c = 1,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT_CZ1d, self).__init__()
       
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
       
        self.A = sparseKernelFT1d(k, alpha, c)
        self.B = sparseKernelFT1d(k, alpha, c)
        self.C = sparseKernelFT1d(k, alpha, c)
       
        self.T0 = nn.Linear(k, k)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))
       
        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))
       
       
    def forward(self, x):
       
        B, N, c, ich = x.shape # (B, N, k)
        ns = math.floor(np.log2(N))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
#         decompose
        for i in range(ns-self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x) # coarsest scale transform

#        reconstruct           
        for i in range(ns-1-self.L,-1,-1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        return x

   
    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                       ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s
       
       
    def evenOdd(self, x):
       
        B, N, c, ich = x.shape # (B, N, c, k)
        assert ich == 2*self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)
       
        x = torch.zeros(B, N*2, c, self.k,
            device = x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x
   
   
class MWT1d(nn.Module):
    def __init__(self,
                 ich = 1, k = 3, alpha = 2, c = 1,
                 nCZ = 3,
                 L = 0,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT1d,self).__init__()
       
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk = nn.Linear(ich, c*k)
       
        self.MWT_CZ = nn.ModuleList(
            [MWT_CZ1d(k, alpha, L, c, base,
            initializer) for _ in range(nCZ)]
        )
        self.Lc0 = nn.Linear(c*k, 128)
        self.Lc1 = nn.Linear(128, 1)
       
        if initializer is not None:
            self.reset_parameters(initializer)
       
    def forward(self, x):
       
        B, N, ich = x.shape # (B, N, d)
        ns = math.floor(np.log2(N))
        x = self.Lk(x)
        x = x.view(B, N, self.c, self.k)
   
        for i in range(self.nCZ):
            x = self.MWT_CZ[i](x)
            if i < self.nCZ-1:
                x = F.relu(x)

        x = x.view(B, N, -1) # collapse c and k
        x = self.Lc0(x)
        x = F.relu(x)
        x = self.Lc1(x)
        return x.squeeze()
   
    def reset_parameters(self, initializer):
        initializer(self.Lc0.weight)
        initializer(self.Lc1.weight)
        
        

class sparseKernel2d(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernel2d,self).__init__()
        
        self.k = k
        self.conv = self.convBlock(k, c*k**2, alpha)
        self.Lo = nn.Linear(alpha*k**2, c*k**2)
        
    def forward(self, x):
        B, Nx, Ny, c, ich = x.shape # (B, Nx, Ny, c, k**2)
        x = x.view(B, Nx, Ny, -1)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, c, ich)
        
        return x
        
        
    def convBlock(self, k, W, alpha):
        och = alpha * k**2
        net = nn.Sequential(
            nn.Conv2d(W, och, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        return net 
    
    
def compl_mul2d(x, weights):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", x, weights)


class sparseKernelFT2d(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernelFT2d, self).__init__()        
        
        self.modes = alpha

        self.weights1 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, dtype=torch.cfloat))        
        nn.init.xavier_normal_(self.weights1)
        nn.init.xavier_normal_(self.weights2)
        
        self.Lo = nn.Linear(c*k**2, c*k**2)
        self.k = k
        
    def forward(self, x):
        B, Nx, Ny, c, ich = x.shape # (B, N, N, c, k^2)
        
        x = x.view(B, Nx, Ny, -1)
        x = x.permute(0, 3, 1, 2)
        x_fft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        l1 = min(self.modes, Nx//2+1)
        l1l = min(self.modes, Nx//2-1)
        l2 = min(self.modes, Ny//2+1)
        out_ft = torch.zeros(B, c*ich, Nx, Ny//2 + 1,  device=x.device, dtype=torch.cfloat)
        
        out_ft[:, :, :l1, :l2] = compl_mul2d(
            x_fft[:, :, :l1, :l2], self.weights1[:, :, :l1, :l2])
        out_ft[:, :, -l1:, :l2] = compl_mul2d(
                x_fft[:, :, -l1:, :l2], self.weights2[:, :, :l1, :l2])
        
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s = (Nx, Ny))
        
        x = x.permute(0, 2, 3, 1)
        x = F.relu(x)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, c, ich)
        return x
        
    
class MWT_CZ2d(nn.Module):
    def __init__(self,
                 k = 3, alpha = 5, 
                 L = 0, c = 1,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT_CZ2d, self).__init__()
        
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

        self.A = sparseKernelFT2d(k, alpha, c)
        self.B = sparseKernel2d(k, c, c)
        self.C = sparseKernel2d(k, c, c)
        
        self.T0 = nn.Linear(c*k**2, c*k**2)

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
        
        B, Nx, Ny, c, ich = x.shape # (B, Nx, Ny, c, k**2)
        ns = math.floor(np.log2(Nx))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

#         decompose
        for i in range(ns-self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x.view(B, 2**self.L, 2**self.L, -1)).view(
            B, 2**self.L, 2**self.L, c, ich) # coarsest scale transform

#        reconstruct            
        for i in range(ns-1-self.L,-1,-1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)

        return x

    
    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2 , ::2 , :, :], 
                        x[:, ::2 , 1::2, :, :], 
                        x[:, 1::2, ::2 , :, :], 
                        x[:, 1::2, 1::2, :, :]
                       ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s
        
        
    def evenOdd(self, x):
        
        B, Nx, Ny, c, ich = x.shape # (B, Nx, Ny, c, k**2)
        assert ich == 2*self.k**2
        x_ee = torch.matmul(x, self.rc_ee)
        x_eo = torch.matmul(x, self.rc_eo)
        x_oe = torch.matmul(x, self.rc_oe)
        x_oo = torch.matmul(x, self.rc_oo)
        
        x = torch.zeros(B, Nx*2, Ny*2, c, self.k**2, 
            device = x.device)
        x[:, ::2 , ::2 , :, :] = x_ee
        x[:, ::2 , 1::2, :, :] = x_eo
        x[:, 1::2, ::2 , :, :] = x_oe
        x[:, 1::2, 1::2, :, :] = x_oo
        return x
    
    def reset_parameters(self, initializer):
        initializer(self.T0.weight)
    
    
class MWT2d(nn.Module):
    def __init__(self,
                 ich = 1, k = 3, alpha = 2, c = 1,
                 nCZ = 3,
                 L = 0,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT2d,self).__init__()
        
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk = nn.Linear(ich, c*k**2)
        
        self.MWT_CZ = nn.ModuleList(
            [MWT_CZ2d(k, alpha, L, c, base, 
            initializer) for _ in range(nCZ)]
        )
        self.Lc0 = nn.Linear(c*k**2, 128)
        self.Lc1 = nn.Linear(128, 1)
        
        if initializer is not None:
            self.reset_parameters(initializer)
        
    def forward(self, x):
        
        B, Nx, Ny, ich = x.shape # (B, Nx, Ny, d)
        ns = math.floor(np.log2(Nx))
        x = self.Lk(x)
        x = x.view(B, Nx, Ny, self.c, self.k**2)
    
        for i in range(self.nCZ):
            x = self.MWT_CZ[i](x)
            if i < self.nCZ-1:
                x = F.relu(x)

        x = x.view(B, Nx, Ny, -1) # collapse c and k**2
        x = self.Lc0(x)
        x = F.relu(x)
        x = self.Lc1(x)
        return x.squeeze()
    
    def reset_parameters(self, initializer):
        initializer(self.Lc0.weight)
        initializer(self.Lc1.weight)
        
        
class sparseKernel(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernel,self).__init__()
        
        self.k = k
        self.conv = self.convBlock(k, c*k**2, alpha)
        self.Lo = nn.Linear(alpha*k**2, c*k**2)
        
    def forward(self, x):
        B, Nx, Ny, c, ich = x.shape # (B, Nx, Ny, c, k**2)
        x = x.view(B, Nx, Ny, -1)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, c, ich)
        
        return x
        
        
    def convBlock(self, k, W, alpha):
        och = alpha * k**2
        net = nn.Sequential(
            nn.Conv2d(W, och, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        return net 
    
    
class sparseKernel3d(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernel3d,self).__init__()
        
        self.k = k
        self.conv = self.convBlock(alpha*k**2, alpha*k**2)
        self.Lo = nn.Linear(alpha*k**2, c*k**2)
        
    def forward(self, x):
        B, Nx, Ny, T, c, ich = x.shape # (B, Nx, Ny, T, c, k**2)
        x = x.view(B, Nx, Ny, T, -1)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, T, c, ich)
        
        return x
        
        
    def convBlock(self, ich, och):
        net = nn.Sequential(
            nn.Conv3d(ich, och, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        return net 
    
    
def compl_mul3d(input, weights):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixyz,ioxyz->boxyz", input, weights)


class sparseKernelFT3d(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernelFT3d, self).__init__()        
        
        self.modes = alpha

        self.weights1 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, dtype=torch.cfloat))        
        self.weights3 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, dtype=torch.cfloat))        
        self.weights4 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, dtype=torch.cfloat))        
        nn.init.xavier_normal_(self.weights1)
        nn.init.xavier_normal_(self.weights2)
        nn.init.xavier_normal_(self.weights3)
        nn.init.xavier_normal_(self.weights4)
        
        self.Lo = nn.Linear(c*k**2, c*k**2)
        self.k = k
        
    def forward(self, x):
        B, Nx, Ny, T, c, ich = x.shape # (B, N, N, T, c, k^2)
        
        x = x.view(B, Nx, Ny, T, -1)
        x = x.permute(0, 4, 1, 2, 3)
        x_fft = torch.fft.rfftn(x, dim = [-3, -2, -1])
        
        # Multiply relevant Fourier modes
        l1 = min(self.modes, Nx//2+1)
        l2 = min(self.modes, Ny//2+1)
        out_ft = torch.zeros(B, c*ich, Nx, Ny, T//2 +1, device=x.device, dtype=torch.cfloat)
        
        out_ft[:, :, :l1, :l2, :self.modes] = compl_mul3d(
            x_fft[:, :, :l1, :l2, :self.modes], self.weights1[:, :, :l1, :l2, :])
        out_ft[:, :, -l1:, :l2, :self.modes] = compl_mul3d(
                x_fft[:, :, -l1:, :l2, :self.modes], self.weights2[:, :, :l1, :l2, :])
        out_ft[:, :, :l1, -l2:, :self.modes] = compl_mul3d(
                x_fft[:, :, :l1, -l2:, :self.modes], self.weights3[:, :, :l1, :l2, :])
        out_ft[:, :, -l1:, -l2:, :self.modes] = compl_mul3d(
                x_fft[:, :, -l1:, -l2:, :self.modes], self.weights4[:, :, :l1, :l2, :])
        
        #Return to physical space
        x = torch.fft.irfftn(out_ft, s = (Nx, Ny, T))
        
        x = x.permute(0, 2, 3, 4, 1)
        x = F.relu(x)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, T, c, ich)
        return x
        
    
class MWT_CZ3d(nn.Module):
    def __init__(self,
                 k = 3, alpha = 5, 
                 L = 0, c = 1, shape_multiplier = 1,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT_CZ3d, self).__init__()
        
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

        self.A = sparseKernelFT3d(k, alpha, c)
        self.B = sparseKernel3d(k, c, c)
        self.C = sparseKernel3d(k, c, c)
        

        self.T0 = nn.Linear(c*k**2*shape_multiplier, c*k**2)

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
        B, Nx, Ny, T, c, ich = x.shape # (B, Nx, Ny, T, c, k**2)
        ns = math.floor(np.log2(Nx))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

#         decompose
        for i in range(ns-self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        
        # Debug: print tensor shapes before T0
        
        x_reshaped = x.view(B, 2**self.L, 2**self.L, T, -1)
        # print(f"DEBUG: x shape before T0: {x_reshaped.shape}")
        # print(f"DEBUG: T0 expects input dim: {self.T0.in_features}, T0 output dim: {self.T0.out_features}")
        # print(f"DEBUG: c={c}, ich={ich}, c*ich={c*ich}")
        
        x = self.T0(x_reshaped).view(
            B, 2**self.L, 2**self.L, T, c, ich) # coarsest scale transform

#        reconstruct            
        for i in range(ns-1-self.L,-1,-1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)

        return x

    
    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2 , ::2 , :, :, :], 
                        x[:, ::2 , 1::2, :, :, :], 
                        x[:, 1::2, ::2 , :, :, :], 
                        x[:, 1::2, 1::2, :, :, :]
                       ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s
        
        
    def evenOdd(self, x):
        
        B, Nx, Ny, T, c, ich = x.shape # (B, Nx, Ny, c, k**2)
        assert ich == 2*self.k**2
        x_ee = torch.matmul(x, self.rc_ee)
        x_eo = torch.matmul(x, self.rc_eo)
        x_oe = torch.matmul(x, self.rc_oe)
        x_oo = torch.matmul(x, self.rc_oo)
        
        x = torch.zeros(B, Nx*2, Ny*2, T, c, self.k**2, 
            device = x.device)
        x[:, ::2 , ::2 , :, :, :] = x_ee
        x[:, ::2 , 1::2, :, :, :] = x_eo
        x[:, 1::2, ::2 , :, :, :] = x_oe
        x[:, 1::2, 1::2, :, :, :] = x_oo
        return x
    
    def reset_parameters(self, initializer):
        initializer(self.T0.weight)
    
    
class MWT3d(nn.Module):
    def __init__(self,
                 k = 3, alpha = 2, c = 1,
                 nCZ = 3,
                 L = 0,
                 base = 'legendre',
                 initializer = None,
                 shape_in = None,
                 shape_out = None,
                 **kwargs):
        super(MWT3d,self).__init__()
        
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ

        self.shape_in = shape_in
        self.shape_out = shape_out
        self.dim_in = shape_in[-1]
        self.dim_out = shape_out[-1] * shape_out[0] // shape_in[0] # C_out * T_out / T_in
        shape_multiplier = shape_in[2] // shape_in[1]
        self.Lk = nn.Linear(self.dim_in, c*k**2)

        self.MWT_CZ = nn.ModuleList(
            [MWT_CZ3d(k, alpha, L, c, shape_multiplier, base, 
            initializer) for _ in range(nCZ)]
        )

        self.BN = nn.ModuleList(
            [nn.BatchNorm3d(c*k**2) for _ in range(nCZ)]
        )
        self.Lc0 = nn.Linear(c*k**2, 128)
        self.Lc1 = nn.Linear(128, self.dim_out)
        
        if initializer is not None:
            self.reset_parameters(initializer)
        
    def forward(self, x):
        switch = False
        if x.dim() == 5 and x.shape[-1] < x.shape[1]:
            switch = True
            x = x.permute(0, 2, 3, 1, 4)  # (B, T, H, W, C) -> (B, H, W, T, C)
    
        B, Nx, Ny, T, ich = x.shape # (B, Nx, Ny, T, d)
        ns = math.floor(np.log2(Nx))
        x = self.Lk(x)
        x = x.view(B, Nx, Ny, T, self.c, self.k**2)
        # x = x.permute(0, 4, 5, 1, 2, 3)
        for i in range(self.nCZ):
            x = self.MWT_CZ[i](x)
            if i < self.nCZ-1:
                x = F.relu(x)


        x = x.view(B, Nx, Ny, T, -1) # collapse c and k**2
        x = self.Lc0(x)
        x = F.relu(x)
        x = self.Lc1(x)
        # .squeeze()  # B, Nx, Ny, T, n_targets

        x = x.reshape(*x.shape[:-1], self.shape_out[-1], self.shape_out[0] // self.shape_in[0]) 
        # (B, Nx, Ny, T, C_out, T_out / T_in) -> (B, T_out, H, W, C_out)
        if switch:
            x = x.permute(0, 3, 5, 1, 2, 4).reshape(x.shape[0], *self.shape_out)

        return x
    
    def train_loss(self, input, target):
        pred = self.forward(input)
        return mse_loss(pred, target)

    def reset_parameters(self, initializer):
        initializer(self.Lc0.weight)
        initializer(self.Lc1.weight)

    def load_checkpoint(self, checkpoint_path, device='cpu'):
        """
        Load pre-trained checkpoint with shape and key matching.

        Args:
            checkpoint_path: Path to the saved checkpoint file
            device: Device to load the model on (e.g., 'cpu' or 'cuda')
        """
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        # Extract state_dict from common formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model_state = self.state_dict()
        compatible_state = {}
        incompatible_layers = []

        for key, param in state_dict.items():
            if key in model_state:
                if param.shape == model_state[key].shape:
                    compatible_state[key] = param.to(device)
                else:
                    incompatible_layers.append(f"{key}: checkpoint {param.shape} vs model {model_state[key].shape}")
            else:
                incompatible_layers.append(f"{key}: not found in model")

        # Report
        total = len(state_dict)
        loaded = len(compatible_state)
        print(f"Compatible parameters: {loaded}/{total} ({100*loaded/total:.1f}%)")

        if incompatible_layers:
            print("Incompatible layers:")
            for l in incompatible_layers[:5]:
                print(f"  - {l}")
            if len(incompatible_layers) > 5:
                print(f"  ... and {len(incompatible_layers) - 5} more")

        self.load_state_dict(compatible_state, strict=False)
        print("Checkpoint loaded successfully.")
