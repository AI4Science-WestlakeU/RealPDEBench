from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_

from realpdebench.model.model import Model as BaseModel


_ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU, 'silu': nn.SiLU,
}


def _act(name):
    cls = _ACTIVATION[name]
    return cls() if isinstance(cls, type) else cls


def _unified_pos_2d(H, W, ref):
    """Distance-based 2D positional encoding. Returns (H*W, ref²)."""
    gh = torch.linspace(0, 1, H)
    gw = torch.linspace(0, 1, W)
    grid = torch.stack(torch.meshgrid(gh, gw, indexing='ij'), dim=-1)
    rh = torch.linspace(0, 1, ref)
    rw = torch.linspace(0, 1, ref)
    grid_ref = torch.stack(torch.meshgrid(rh, rw, indexing='ij'), dim=-1)
    dist = torch.sqrt(torch.sum(
        (grid[:, :, None, None, :] - grid_ref[None, None, :, :, :]) ** 2, dim=-1))
    return dist.reshape(H * W, ref ** 2)


class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, act='gelu'):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hid), _act(act), nn.Linear(n_hid, n_out))

    def forward(self, x):
        return self.net(x)


class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super().__init__()
        self.w_gate = nn.Linear(dim_in, dim_hid)
        self.w_up = nn.Linear(dim_in, dim_hid)
        self.w_down = nn.Linear(dim_hid, dim_out)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TemporalAggregator(nn.Module):
    """Pool time using a projected temporal mean as the query."""

    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)

    def forward(self, x):  # (B, N, T, C)
        q = self.query_proj(x.mean(dim=2))  # (B, N, C)
        k = self.key_proj(x)  # (B, N, T, C)
        attn = torch.einsum('bnc,bntc->bnt', q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        return torch.einsum('bnt,bntc->bnc', attn, x)


class PhysicsAttention2D(nn.Module):
    """DW-Sep (5×5 DW + 1×1 PW) 2D slice attention. Input: (B_eff, N=H*W, C)."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32, H=64, W=128):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.H, self.W = H, W

        def _dwsep2d(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, (5, 5), 1, (2, 2), groups=in_ch),
                nn.Conv2d(in_ch, out_ch, 1),
            )

        self.in_project_x = _dwsep2d(dim, inner_dim)
        self.in_project_fx = _dwsep2d(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.gamma = nn.Parameter(torch.zeros(1))

    def _project(self, proj, x, B_eff, N):
        H, W = self.H, self.W
        x2d = x.reshape(B_eff, H, W, x.shape[-1]).permute(0, 3, 1, 2).contiguous()
        out = proj(x2d)
        return (out.permute(0, 2, 3, 1).contiguous()
                   .reshape(B_eff, N, self.heads, self.dim_head)
                   .permute(0, 2, 1, 3).contiguous())

    def forward(self, x):
        B_eff, N, C = x.shape
        fx_mid = self._project(self.in_project_fx, x, B_eff, N)
        x_mid = self._project(self.in_project_x,  x, B_eff, N)
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, 0.1, 5))
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5).unsqueeze(-1))
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.softmax(dots))
        out_slice = torch.matmul(attn, v)
        out_slice = self.gamma * out_slice + slice_token
        out_x = torch.einsum('bhgc,bhng->bhnc', out_slice, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class _AxisSpectralConv3D(nn.Module):
    def __init__(self, dim, modes, axis):
        super().__init__()
        assert axis in ('D', 'H', 'W')
        self.modes = modes
        self.axis = axis
        scale = 1.0 / (dim * dim)
        self.weights = nn.Parameter(scale * torch.randn(dim, dim, modes, 2))

    def forward(self, x):  # (B, D, H, W, C)
        B, D, H, W, C = x.shape
        w_c = torch.view_as_complex(self.weights.contiguous())
        if self.axis == 'D':
            xp = x.permute(0, 4, 2, 3, 1).contiguous()
            n = D
            xft = torch.fft.rfft(xp, norm='ortho')
            fm = min(self.modes, n // 2 + 1)
            out_ft = torch.zeros(B, C, H, W, n // 2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[..., :fm] = torch.einsum('bchwm,com->bohwm', xft[..., :fm], w_c[:, :, :fm])
            out = torch.fft.irfft(out_ft, n=n, norm='ortho')
            return out.permute(0, 4, 2, 3, 1).contiguous()
        elif self.axis == 'H':
            xp = x.permute(0, 4, 1, 3, 2).contiguous()
            n = H
            xft = torch.fft.rfft(xp, norm='ortho')
            fm = min(self.modes, n // 2 + 1)
            out_ft = torch.zeros(B, C, D, W, n // 2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[..., :fm] = torch.einsum('bcdwm,com->bodwm', xft[..., :fm], w_c[:, :, :fm])
            out = torch.fft.irfft(out_ft, n=n, norm='ortho')
            return out.permute(0, 2, 4, 3, 1).contiguous()
        else:
            xp = x.permute(0, 4, 1, 2, 3).contiguous()
            n = W
            xft = torch.fft.rfft(xp, norm='ortho')
            fm = min(self.modes, n // 2 + 1)
            out_ft = torch.zeros(B, C, D, H, n // 2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[..., :fm] = torch.einsum('bcdhm,com->bodhm', xft[..., :fm], w_c[:, :, :fm])
            out = torch.fft.irfft(out_ft, n=n, norm='ortho')
            return out.permute(0, 2, 3, 4, 1).contiguous()


class FactorizedSpectralConv3D(nn.Module):
    def __init__(self, dim, modes_d=8, modes_h=16, modes_w=16):
        super().__init__()
        self.spec_d = _AxisSpectralConv3D(dim, modes_d, 'D')
        self.spec_h = _AxisSpectralConv3D(dim, modes_h, 'H')
        self.spec_w = _AxisSpectralConv3D(dim, modes_w, 'W')

    def forward(self, x):  # (B, D, H, W, C)
        return self.spec_d(x) + self.spec_h(x) + self.spec_w(x)


class _AxisSpectralConv2D(nn.Module):
    def __init__(self, dim, modes, axis, H, W):
        super().__init__()
        assert axis in ('H', 'W')
        self.modes = modes
        self.axis = axis
        self.H, self.W = H, W
        scale = 1.0 / (dim * dim)
        self.weights = nn.Parameter(scale * torch.randn(dim, dim, modes, 2))

    def forward(self, x):  # (B_eff, H, W, C)
        B, H, W, C = x.shape
        w_c = torch.view_as_complex(self.weights.contiguous())
        if self.axis == 'H':
            xp = x.permute(0, 3, 2, 1).contiguous()
            n = H
            xft = torch.fft.rfft(xp, norm='ortho')
            fm = min(self.modes, n // 2 + 1)
            out_ft = torch.zeros(B, C, W, n // 2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[..., :fm] = torch.einsum('bcwm,com->bowm', xft[..., :fm], w_c[:, :, :fm])
            out = torch.fft.irfft(out_ft, n=n, norm='ortho')
            return out.permute(0, 3, 2, 1).contiguous()
        else:
            xp = x.permute(0, 3, 1, 2).contiguous()
            n = W
            xft = torch.fft.rfft(xp, norm='ortho')
            fm = min(self.modes, n // 2 + 1)
            out_ft = torch.zeros(B, C, H, n // 2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[..., :fm] = torch.einsum('bchm,com->bohm', xft[..., :fm], w_c[:, :, :fm])
            out = torch.fft.irfft(out_ft, n=n, norm='ortho')
            return out.permute(0, 2, 3, 1).contiguous()


class FactorizedSpectralConv2D(nn.Module):
    def __init__(self, dim, modes, H, W, modes_w=None):
        super().__init__()
        self.spec_h = _AxisSpectralConv2D(dim, modes, 'H', H, W)
        self.spec_w = _AxisSpectralConv2D(dim, modes_w if modes_w is not None else modes, 'W', H, W)

    def forward(self, x):  # (B_eff, H, W, C)
        return self.spec_h(x) + self.spec_w(x)


class SpatiotemporalBlock(nn.Module):
    """Spatiotemporal CPE followed by parallel spectral and slice attention branches."""

    def __init__(self, num_heads, hidden_dim, dropout=0., act='gelu', mlp_ratio=2,
                 D=10, H=64, W=128, spectral_modes_d=8, spectral_modes_hw=16,
                 spectral_modes_w=None, slice_num=32):
        super().__init__()
        self.D, self.H, self.W = D, H, W
        half = hidden_dim // 2
        dh = half // num_heads

        self.cpe = nn.Conv3d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)

        self.ln_spec = nn.LayerNorm(half)
        self.spectral = FactorizedSpectralConv3D(
            half, modes_d=spectral_modes_d,
            modes_h=spectral_modes_hw,
            modes_w=spectral_modes_w if spectral_modes_w is not None else spectral_modes_hw)

        self.ln_1 = nn.LayerNorm(half)
        self.Attn = PhysicsAttention2D(
            dim=half, heads=num_heads, dim_head=dh,
            dropout=dropout, slice_num=slice_num, H=H, W=W)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = SwiGLU(hidden_dim, int(hidden_dim * mlp_ratio * 2 / 3), hidden_dim)

    def forward(self, fx):  # (B, N=D*H*W, C)
        B, N, C = fx.shape
        D, H, W = self.D, self.H, self.W
        half = C // 2

        fx3d = fx.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        fx = fx + self.cpe(fx3d).permute(0, 2, 3, 4, 1).reshape(B, N, C)

        x_spec, x_phys = fx.chunk(2, dim=-1)

        spec_out = x_spec + self.spectral(
            self.ln_spec(x_spec).reshape(B, D, H, W, half)).reshape(B, N, half)

        x_phys_pf = self.ln_1(x_phys).reshape(B * D, H * W, half)
        phys_out = x_phys + self.Attn(x_phys_pf).reshape(B, N, half)

        fused = torch.cat([spec_out, phys_out], dim=-1)
        fx = self.mlp(self.ln_2(fused)) + fused
        return fx


class SpatialBlock(nn.Module):
    """Spatial CPE and parallel branches, with an optional output projection."""

    def __init__(self, num_heads, hidden_dim, dropout=0., act='gelu', mlp_ratio=2,
                 last_out_dim=0, H=64, W=128, spectral_modes_hw=16,
                 spectral_modes_w=None, slice_num=32):
        super().__init__()
        self.last_out_dim = last_out_dim
        self.H, self.W = H, W
        half = hidden_dim // 2
        dh = half // num_heads

        self.cpe = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)

        self.ln_spec = nn.LayerNorm(half)
        self.spectral = FactorizedSpectralConv2D(
            half, modes=spectral_modes_hw, H=H, W=W, modes_w=spectral_modes_w)

        self.ln_1 = nn.LayerNorm(half)
        self.Attn = PhysicsAttention2D(
            dim=half, heads=num_heads, dim_head=dh,
            dropout=dropout, slice_num=slice_num, H=H, W=W)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = SwiGLU(hidden_dim, int(hidden_dim * mlp_ratio * 2 / 3), hidden_dim)

        if last_out_dim > 0:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, last_out_dim)

    def forward(self, fx):  # (B_eff, N=H*W, C)
        B_eff, N, C = fx.shape
        H, W = self.H, self.W
        half = C // 2

        fx2d = fx.reshape(B_eff, H, W, C).permute(0, 3, 1, 2).contiguous()
        fx = fx + self.cpe(fx2d).permute(0, 2, 3, 1).reshape(B_eff, N, C)

        x_spec, x_phys = fx.chunk(2, dim=-1)
        spec_out = x_spec + self.spectral(
            self.ln_spec(x_spec).reshape(B_eff, H, W, half)).reshape(B_eff, N, half)
        phys_out = x_phys + self.Attn(self.ln_1(x_phys))

        fused = torch.cat([spec_out, phys_out], dim=-1)
        fx = self.mlp(self.ln_2(fused)) + fused

        if self.last_out_dim > 0:
            return self.mlp2(self.ln_3(fx))
        return fx


class TransolverSigma(BaseModel):
    """Predict fields with layout ``[batch, time, height, width, channels]``.

    Input/output time and spatial dimensions must match. Channels may differ.
    ``n_layers`` is split between the spatiotemporal and spatial stages.
    """

    def __init__(self, shape_in, shape_out, n_hidden=128, n_head=8, n_layers=8,
                 mlp_ratio=2, slice_num=32,
                 dropout=0., act='gelu',
                 spectral_modes_d=8, spectral_modes_hw=16, spectral_modes_w=None,
                 ref=4):
        super().__init__()
        if len(shape_in) != 4 or len(shape_out) != 4:
            raise ValueError("Expected (time, height, width, channels) shapes")
        if tuple(shape_in[:3]) != tuple(shape_out[:3]):
            raise ValueError("Input and output time/spatial dimensions must match")
        if any(size <= 0 for size in (*shape_in, *shape_out)):
            raise ValueError("All input and output dimensions must be positive")
        if n_layers < 2:
            raise ValueError("n_layers must include at least one block per stage")
        if n_head < 1 or n_hidden < 2 or n_hidden % (2 * n_head):
            raise ValueError("n_hidden must be divisible by 2 * n_head")
        D, H, W, fun_dim = shape_in
        out_dim = shape_out[-1]
        self.input_shape = tuple(shape_in)
        self.D, self.H, self.W = D, H, W
        self.out_dim = out_dim
        n1 = n_layers // 2
        n2 = n_layers - n1

        pos = _unified_pos_2d(H, W, ref)
        self.register_buffer('pos', pos)

        t = torch.linspace(0, 1, D)
        self.register_buffer('t_coord', t)

        self.preprocess = MLP(fun_dim + ref ** 2 + 1, n_hidden * 2, n_hidden, act=act)
        self.placeholder = nn.Parameter((1.0 / n_hidden) * torch.rand(n_hidden))

        self.stage1 = nn.ModuleList([
            SpatiotemporalBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio,
                D=D, H=H, W=W,
                spectral_modes_d=spectral_modes_d,
                spectral_modes_hw=spectral_modes_hw,
                spectral_modes_w=spectral_modes_w,
                slice_num=slice_num,
            )
            for _ in range(n1)
        ])

        self.time_agg = TemporalAggregator(n_hidden)

        self.stage2 = nn.ModuleList([
            SpatialBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio,
                last_out_dim=(D * out_dim if i == n2 - 1 else 0),
                H=H, W=W,
                spectral_modes_hw=spectral_modes_hw,
                spectral_modes_w=spectral_modes_w,
                slice_num=slice_num,
            )
            for i in range(n2)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if x.ndim != 5 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected [batch, {self.input_shape}], got {tuple(x.shape)}")
        B, D, H, W, C_in = x.shape
        N2 = H * W

        x_flat = x.reshape(B * D, N2, C_in)  # (B*D, H*W, C_in)
        pos = self.pos.unsqueeze(0).expand(B * D, -1, -1)  # (B*D, H*W, ref²)
        t = self.t_coord.unsqueeze(-1).expand(D, N2)  # (D, H*W)
        t = t.unsqueeze(0).expand(B, -1, -1).reshape(B * D, N2, 1)
        fx = self.preprocess(torch.cat([pos, x_flat, t], dim=-1))  # (B*D, H*W, n_hidden)
        fx = fx + self.placeholder[None, None, :]

        fx = fx.reshape(B, D * N2, -1)
        for block in self.stage1:
            fx = block(fx)  # (B, D*H*W, C)

        fx = fx.reshape(B, D, N2, -1).permute(0, 2, 1, 3)  # (B, H*W, D, C)
        fx = self.time_agg(fx)  # (B, H*W, C)

        for block in self.stage2:
            fx = block(fx)

        return fx.reshape(B, H, W, D, self.out_dim).permute(0, 3, 1, 2, 4).contiguous()

    def train_loss(self, input, target):
        pred = self.forward(input)
        return (pred - target) ** 2
