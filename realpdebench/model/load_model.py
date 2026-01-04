import logging


def load_model(train_dataset, device='cpu', **kwargs):
    model_name = kwargs['model_name']

    input, target = train_dataset[0] # T, S, S, C
    input_shape = input.shape
    output_shape = target.shape

    logging.info(f"Loading model {model_name} with input shape {input_shape} and output shape {output_shape}")
    if model_name == 'fno':
        from realpdebench.model.fno import FNO3d
        model = FNO3d(
            modes1=kwargs['modes1'], 
            modes2=kwargs['modes2'], 
            modes3=kwargs['modes3'], 
            n_layers=kwargs['n_layers'],
            width=kwargs['width'], 
            shape_in=input_shape, 
            shape_out=output_shape
            ).to(device)

    elif model_name == 'wdno':
        from realpdebench.model.wdno_libs.unet import Unet3d
        from realpdebench.model.wdno import WDNO
        base_model = Unet3d(
            dim = kwargs['dim'],
            dim_mults = kwargs['dim_mults'],
            channels = 8 * (input_shape[-1] + output_shape[-1] * output_shape[0] 
                        // input_shape[0]),
            ).to(device)

        model = WDNO(
                base_model,
                train_dataset,
                kwargs['dataset_root'],
                kwargs['dataset_name'],
                kwargs['wave_type'],
                kwargs['pad_mode'],
                timesteps = 1000, 
                beta_schedule = kwargs['beta_schedule'],
                sampling_timesteps = kwargs['sampling_timesteps'],   
                ddim_sampling_eta = kwargs['ddim_sampling_eta'],
            ).to(device)
    
    elif model_name == 'unet':
        # input: [bz, t, x, y, c_in]
        # output: [bz, t, x, y, c_out]
        from realpdebench.model.unet import Unet3d
        model = Unet3d(
                    dim=input_shape[1], 
                    out_channels=output_shape[-1],
                    dim_mults=kwargs['dim_mults'], 
                    channels=input_shape[-1],
                    in_time=input_shape[0],
                    out_time=output_shape[0],
                    ).to(device)
        
    elif model_name == 'cno':
        from realpdebench.model.cno import CNO3d
        if output_shape[0] > input_shape[0] and output_shape[0] % input_shape[0] == 0:
            out_dim_mult = output_shape[0] // input_shape[0]
        elif output_shape[0] == input_shape[0]:
            out_dim_mult = 1
        else:
            raise ValueError(f"Output shape {output_shape[1]} is not a multiple of input shape {input_shape[1]}")
        
        model = CNO3d(
                        in_dim=input_shape[-1],
                        out_dim=output_shape[-1],
                        out_dim_mult=out_dim_mult,
                        in_size=input_shape[2],
                        N_layers=kwargs['N_layers'],
                    ).to(device)
        
    elif model_name == 'galerkin_transformer':
        from realpdebench.model.galerkin_transformer import GalerkinTransformer3d
        kwargs['node_feats'] = input_shape[-1]
        kwargs['n_targets'] = output_shape[-1]
        kwargs['shape_in'] = input_shape
        kwargs['shape_out'] = output_shape
        
        if 'config' in kwargs:
            kwargs.pop('config')

        # print("Debug: kwargs =", kwargs)
        # print("config type:", type(kwargs))
        # print("config keys:", kwargs.keys()) 
        
        model = GalerkinTransformer3d(**kwargs).to(device)

    elif model_name == 'mwt':
        from realpdebench.model.MWT_libs.models import MWT3d
        # from model.MWT_libs.models_fix import MWT3d
        
        kwargs['shape_in'] = input_shape
        kwargs['shape_out'] = output_shape
        
        if 'config' in kwargs:
            kwargs.pop('config')

        print("Debug: kwargs =", kwargs)
        print("config type:", type(kwargs))
        print("config keys:", kwargs.keys()) 
        
        model = MWT3d(**kwargs).to(device)
    elif model_name == 'dpot':
        from realpdebench.model.dpot import DPOT
        model = DPOT(
            shape_in=input_shape,
            shape_out=output_shape,
            img_size=kwargs["img_size"],
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["out_channels"],
            in_timesteps=kwargs["in_timesteps"],
            out_timesteps=kwargs["out_timesteps"],
            patch_size=kwargs["patch_size"],
            embed_dim=kwargs["embed_dim"],
            depth=kwargs["depth"],
            n_blocks=kwargs["n_blocks"],
            modes=kwargs["modes"],
            mlp_ratio=kwargs["mlp_ratio"],
            out_layer_dim=kwargs["out_layer_dim"],
            normalize=kwargs["normalize"],
            act=kwargs["act"],
            time_agg=kwargs["time_agg"],
            n_cls=kwargs["n_cls"],
            model_type=kwargs["model_type"],
            checkpoint_path=kwargs["checkpoint_path"],
        ).to(device)
    elif model_name == 'deeponet':
        from realpdebench.model.deeponet import DeepONet
        
        model = DeepONet(
            shape_in=input_shape,
            shape_out=output_shape,
            input_channels=input_shape[-1],
            output_channels=output_shape[-1],
            p=kwargs['p'],
            dropout_rate=kwargs['dropout_rate'],
            device=device
        ).to(device)

    elif model_name == 'transolver':
        from realpdebench.model.TRANSOLVER_libs.Transolver_Structured_Mesh_3D import Model
        
        model = Model(
            space_dim=kwargs['space_dim'], n_layers=kwargs['n_layers'], n_hidden=kwargs['n_hidden'], n_head=kwargs['n_head'],
            H=kwargs['H'], W=kwargs['W'], D=kwargs['D'], Time_Input=False, unified_pos=False,
            fun_dim=kwargs['fun_dim'], out_dim=kwargs['out_dim'], ref=kwargs['ref'],dropout=kwargs['dropout'], act=kwargs['act'], mlp_ratio=kwargs['mlp_ratio'], slice_num=kwargs['slice_num']
        ).to(device)
    
    elif model_name == 'dmd':
        from realpdebench.model.dmd import DMD
        model = DMD(n_modes=kwargs['n_modes'], n_predict=kwargs['n_predict'], input_feature=kwargs['input_feature'], n_autoregressive=kwargs['N_autoregressive'])
        # model = dmd_model.dmd_run

    else:
        raise ValueError(f"Model {model_name} not supported")
    return model