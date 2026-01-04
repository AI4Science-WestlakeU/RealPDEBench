#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import numpy as np
import torch
import torch.nn as nn
import time
import pandas as pd
from typing import Sequence
from einops import rearrange
from collections import OrderedDict




class MultipleTensors(Sequence):
    def __init__(self, x):
        self.x = x

    def to(self, device):
        self.x = [x_.to(device) for x_ in self.x]
        return self

    def __len__(self):
        return len(self.x)

    def numel(self):
        return np.sum([x_.numel() for x_ in self.x])


    def __getitem__(self, item):
        return self.x[item]



def get_grid(data, n_dim, multi_channel):
    if n_dim == 1:
        grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]))
        grid = torch.unsqueeze(grid[0], dim=-1)
    elif n_dim == 2:
        grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]))
        grid = torch.stack(grid, dim=-1)
    elif n_dim == 3:
        grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]),torch.linspace(0,1, data.shape[3]))
        grid = torch.stack(grid, dim=-1)
    elif n_dim == 4:
        grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]),torch.linspace(0,1, data.shape[3]),torch.linspace(0,1, data.shape[4]))
        grid = torch.stack(grid, dim=-1)
    else:
        raise NotImplementedError
    grid = grid.to(data.device)
    if multi_channel:
        grid = grid.unsqueeze(-2)
        data = torch.cat([torch.tile(grid.unsqueeze(0), [data.shape[0]] + [1] * n_dim + [data.shape[-2], 1]), data],dim=-1)
    else:
        data = torch.cat([torch.tile(grid.unsqueeze(0), [data.shape[0]] + [1] * n_dim + [1]), data], dim=-1)

    return data



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time for {func.__name__}: {elapsed_time:.5f} seconds")
        return result
    return wrapper


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = 2 * parameter.numel() if parameter.is_complex() else parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def load_model_from_checkpoint(model, model_state_dict):
    if next(iter(model_state_dict.keys())).startswith('module.'):
        new_state_dict = OrderedDict()
        for key, item in model_state_dict.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = item
        del model_state_dict
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    return


def load_components_from_pretrained(model, state_dict, components='all'):
    """
    :model: the model
    :state_dict: state_dict of source model
    :components: 'all' or list from 'patch', 'pos', 'blocks','time_agg','cls_head', 'scale_feats', 'out'
    """

    if next(iter(state_dict.keys())).startswith('module.'):
        new_state_dict = OrderedDict()
        for key, item in state_dict.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = item
        del state_dict
        state_dict = new_state_dict
    if (components == 'all') or ('all' in components):
        model.load_state_dict(state_dict)
        return
    else:
        for name in components:
            if name == 'patch_embed' and hasattr(model, 'patch_embed'):
                model.patch_embed.load_state_dict(OrderedDict(
                    {k.replace('patch_embed.', ''): v for k, v in state_dict.items() if k.startswith('patch_embed.')}))
            elif name == 'pos' and hasattr(model, 'pos_embed'):
                # model.pos_embed.load_state_dict(OrderedDict(
                #     {k.replace('pos_embed.', ''): v for k, v in state_dict.items() if k.startswith('pos_embed.')}))
                model.pos_embed = nn.Parameter(state_dict['pos_embed'])
                # pos_embed_state = OrderedDict({k: v for k, v in state_dict.items() if k.startswith('pos_embed')})
                # if pos_embed_state:
                #     key, value = next(iter(pos_embed_state.items()))
                #     setattr(model, 'pos_embed', value)
            elif name == 'blocks' and hasattr(model, 'blocks'):
                for i, block in enumerate(model.blocks):
                    block_state_dict = OrderedDict({k.replace(f'blocks.{i}.', ''): v for k, v in state_dict.items() if
                                                    k.startswith(f'blocks.{i}.')})
                    block.load_state_dict(block_state_dict)
            elif name == 'scale_feats' and hasattr(model, 'scale_feats_mu'):
                model.scale_feats_mu.load_state_dict(OrderedDict(
                    {k.replace('scale_feats_mu.', ''): v for k, v in state_dict.items() if
                     k.startswith('scale_feats_mu.')}))
                model.scale_feats_sigma.load_state_dict(OrderedDict(
                    {k.replace('scale_feats_sigma.', ''): v for k, v in state_dict.items() if
                     k.startswith('scale_feats_sigma.')}))
            elif name == 'cls_head' and hasattr(model, 'cls_head'):
                model.cls_head.load_state_dict(OrderedDict(
                    {k.replace('cls_head.', ''): v for k, v in state_dict.items() if k.startswith('cls_head.')}))
            elif name == 'time_agg' and hasattr(model, 'time_agg_layer'):
                model.time_agg_layer.load_state_dict(OrderedDict(
                    {k.replace('time_agg_layer.', ''): v for k, v in state_dict.items() if
                     k.startswith('time_agg_layer.')}))
            elif name == 'out' and hasattr(model, 'out_layer'):
                model.out_layer.load_state_dict(OrderedDict(
                    {k.replace('out_layer.', ''): v for k, v in state_dict.items() if k.startswith('out_layer.')}))
            else:
                print(f"Submodule does not exists：{name}")
        return



def load_3d_components_from_2d(model, state_dict, components='all'):
    """
        :model: the model
        :state_dict: state_dict of source model
        :components: 'all' or list from 'patch', 'pos', 'blocks','time_agg','cls_head', 'scale_feats', 'out'
        """

    if next(iter(state_dict.keys())).startswith('module.'):
        new_state_dict = OrderedDict()
        for key, item in state_dict.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = item
        del state_dict
        state_dict = new_state_dict
    if (components == 'all') or ('all' in components):
        model.load_state_dict(state_dict)
        return
    else:
        for name in components:

            if name == 'blocks' and hasattr(model, 'blocks'):

                for i, block in enumerate(model.blocks):
                    block_state_dict = OrderedDict({k.replace(f'blocks.{i}.', ''): v for k, v in state_dict.items() if
                                                    k.startswith(f'blocks.{i}.')})
                    ## reshape 2d conv param to 3d conv param
                    for k, v in block_state_dict.items():
                        if 'mlp' in k and 'weight' in k:
                            block_state_dict[k] = v.unsqueeze(-1)
                    block.load_state_dict(block_state_dict)

            elif name == 'time_agg' and hasattr(model, 'time_agg_layer'):
                model.time_agg_layer.load_state_dict(OrderedDict(
                    {k.replace('time_agg_layer.', ''): v for k, v in state_dict.items() if
                     k.startswith('time_agg_layer.')}))
            else:
                print(f"Submodule does not exists：{name}")
        return


def save_results_excel(filename, data_dict):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for key, value in data_dict.items():
            df = pd.DataFrame(value)
            df.to_excel(writer, sheet_name=key, index=False)


# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

def samples_fft(u):
    return scipy.fft.fftn(u, s=u.shape[2:], norm='forward', workers=-1)


def samples_ifft(u_hat):
    return scipy.fft.ifftn(u_hat, s=u_hat.shape[2:], norm='forward', workers=-1).real


def downsample(u, N, fourier=False):
    if np.isrealobj(u):
        u_hat = samples_fft(u)
    elif np.iscomplexobj(u):
        u_hat = u
    else:
        raise TypeError(f'Expected either real or complex valued array. Got {u.dtype}.')
    d = u_hat.ndim - 2
    u_hat_down = None
    if d == 2:
        u_hat_down = np.zeros((u_hat.shape[0], u_hat.shape[1], N, N), dtype=u_hat.dtype)
        u_hat_down[:, :, :N // 2 + 1, :N // 2 + 1] = u_hat[:, :, :N // 2 + 1, :N // 2 + 1]
        u_hat_down[:, :, :N // 2 + 1, -N // 2:] = u_hat[:, :, :N // 2 + 1, -N // 2:]
        u_hat_down[:, :, -N // 2:, :N // 2 + 1] = u_hat[:, :, -N // 2:, :N // 2 + 1]
        u_hat_down[:, :, -N // 2:, -N // 2:] = u_hat[:, :, -N // 2:, -N // 2:]
    else:
        raise ValueError(f'Invalid dimension {d}')
    if fourier:
        return u_hat_down
    u_down = samples_ifft(u_hat_down)
    return u_down


def upsample(u, N, fourier=False):
    if np.isrealobj(u):
        u_hat = samples_fft(u)
    elif np.iscomplexobj(u):
        u_hat = u
    else:
        raise TypeError(f'Expected either real or complex valued array. Got {u.dtype}.')
    d = u_hat.ndim - 2
    N_old = u_hat.shape[-2]
    u_hat_up = None
    if d == 2:
        u_hat_up = np.zeros((u_hat.shape[0], u_hat.shape[1], N, N), dtype=u_hat.dtype)
        u_hat_up[:, :, :N_old // 2 + 1, :N_old // 2 + 1] = u_hat[:, :, :N_old // 2 + 1, :N_old // 2 + 1]
        u_hat_up[:, :, :N_old // 2 + 1, -N_old // 2:] = u_hat[:, :, :N_old // 2 + 1, -N_old // 2:]
        u_hat_up[:, :, -N_old // 2:, :N_old // 2 + 1] = u_hat[:, :, -N_old // 2:, :N_old // 2 + 1]
        u_hat_up[:, :, -N_old // 2:, -N_old // 2:] = u_hat[:, :, -N_old // 2:, -N_old // 2:]
    else:
        raise ValueError(f'Invalid dimension {d}')
    if fourier:
        return u_hat_up
    u_up = samples_ifft(u_hat_up)
    return u_up



## B, C, X, Y; B, X, Y, T, C (temporal)
def resize(x, out_size, permute=False, temporal=False):
    if temporal:
        T, C = x.shape[-2:]
        x = rearrange(x, 'b x y t c -> b (t c) x y')
    if permute:
        x = x.permute(0, 3, 1, 2)

    f = torch.fft.rfft2(x, norm='backward')
    f_z = torch.zeros((*x.shape[:-2], out_size[0], out_size[1] // 2 + 1), dtype=f.dtype, device=f.device)
    # 2k+1 -> (2k+1 + 1) // 2 = k+1 and (2k+1)//2 = k
    top_freqs1 = min((f.shape[-2] + 1) // 2, (out_size[0] + 1) // 2)
    top_freqs2 = min(f.shape[-1], out_size[1] // 2 + 1)
    # 2k -> (2k + 1) // 2 = k and (2k)//2 = k
    bot_freqs1 = min(f.shape[-2] // 2, out_size[0] // 2)
    bot_freqs2 = min(f.shape[-1], out_size[1] // 2 + 1)
    f_z[..., :top_freqs1, :top_freqs2] = f[..., :top_freqs1, :top_freqs2]
    f_z[..., -bot_freqs1:, :bot_freqs2] = f[..., -bot_freqs1:, :bot_freqs2]
    # x_z = torch.fft.ifft2(f_z, s=out_size).real
    x_z = torch.fft.irfft2(f_z, s=out_size).real
    x_z = x_z * (out_size[0] / x.shape[-2]) * (out_size[1] / x.shape[-1])

    # f_z[..., -f.shape[-2]//2:, :f.shape[-1]] = f[..., :f.shape[-2]//2+1, :]

    if temporal:
        x_z = rearrange(x_z, 'b (t c) x y -> b x y t c',t=T, c=C)
    if permute:
        x_z = x_z.permute(0, 2, 3, 1)

    return x_z


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # data_dict = {
    #     'a': [[1, 2], [3, 4]],
    #     'b': [[5, 6]]
    # }
    # save_results_excel('test.xlsx', data_dict)
    x = torch.rand(10,2,64,64)
    y = resize(x, [32, 32],permute=False)
    print(y.shape)