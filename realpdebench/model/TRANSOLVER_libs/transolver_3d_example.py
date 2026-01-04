import torch
import torch.nn as nn
import numpy as np
import sys
import os


# 添加模型路径到系统路径
sys.path.append('./realpdebench/realpdebench/model/TRANSOLVER_libs')

# 导入模型相关模块
from Transolver_Structured_Mesh_3D import Model
from Physics_Attention import Physics_Attention_Structured_Mesh_3D
from Embedding import timestep_embedding

def create_transolver_3d_model():
    """
    创建Transolver_Structured_Mesh_3D模型实例
    """
    # 模型参数配置
    model_config = {
        'space_dim': 3,           # 空间维度 (x, y, z)
        'n_layers': 5,            # Transformer层数
        'n_hidden': 256,          # 隐藏层维度
        'dropout': 0.1,           # Dropout率
        'n_head': 8,              # 注意力头数
        'Time_Input': True,       # 是否使用时间输入
        'act': 'gelu',            # 激活函数
        'mlp_ratio': 4,           # MLP扩展比例
        'fun_dim': 1,             # 函数维度
        'out_dim': 1,             # 输出维度
        'slice_num': 32,          # 切片数量
        'ref': 8,                 # 参考网格大小
        'unified_pos': False,     # 是否使用统一位置编码
        'H': 32,                  # 高度
        'W': 32,                  # 宽度
        'D': 32,                  # 深度
    }
    
    # 创建模型
    model = Model(**model_config)
    
    return model, model_config

def generate_sample_data(batch_size=2, H=32, W=32, D=32, space_dim=3, fun_dim=1):
    """
    生成示例数据
    """
    # 生成空间坐标 (x, y, z)
    x_coords = torch.linspace(0, 1, H).reshape(1, H, 1, 1).repeat(batch_size, 1, W, D)
    y_coords = torch.linspace(0, 1, W).reshape(1, 1, W, 1).repeat(batch_size, H, 1, D)
    z_coords = torch.linspace(0, 1, D).reshape(1, 1, 1, D).repeat(batch_size, H, W, 1)
    
    # 组合空间坐标
    spatial_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1)  # [B, H, W, D, 3]
    
    # 生成函数值 (例如：简单的正弦波)
    x_grid, y_grid, z_grid = torch.meshgrid(
        torch.linspace(0, 2*np.pi, H),
        torch.linspace(0, 2*np.pi, W),
        torch.linspace(0, 2*np.pi, D),
        indexing='ij'
    )
    function_values = torch.sin(x_grid + y_grid + z_grid).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, 1, fun_dim)
    
    # 重塑为模型期望的格式
    # 模型期望输入: [batch_size, H*W*D, space_dim] 和 [batch_size, H*W*D, fun_dim]
    x = spatial_coords.reshape(batch_size, H*W*D, space_dim)
    fx = function_values.reshape(batch_size, H*W*D, fun_dim)
    
    # 生成时间步
    T = torch.tensor([0.5, 1.0])  # 两个批次的时间步
    
    return x, fx, T

def main():
    """
    主函数：演示模型的使用
    """
    print("=== Transolver_Structured_Mesh_3D 模型使用示例 ===\n")
    
    # 1. 创建模型
    print("1. 创建模型...")
    model, config = create_transolver_3d_model()
    print(f"模型配置: {config}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 2. 生成示例数据
    print("2. 生成示例数据...")
    batch_size = 2
    x, fx, T = generate_sample_data(
        batch_size=batch_size,
        H=config['H'],
        W=config['W'],
        D=config['D'],
        space_dim=config['space_dim'],
        fun_dim=config['fun_dim']
    )
    
    print(f"输入数据形状:")
    print(f"  x (空间坐标): {x.shape}")
    print(f"  fx (函数值): {fx.shape}")
    print(f"  T (时间步): {T.shape}")
    print()
    
    # 3. 前向传播
    print("3. 执行前向传播...")
    model.eval()  # 设置为评估模式
    
    with torch.no_grad():
        # 使用时间输入
        if config['Time_Input']:
            output = model(x, fx, T)
            print(f"输出形状 (带时间输入): {output.shape}")
        else:
            output = model(x, fx)
            print(f"输出形状 (无时间输入): {output.shape}")
    
    print()
    
    # 4. 演示不同的输入模式
    print("4. 演示不同的输入模式...")
    
    # 模式1: 只有空间坐标，没有函数值
    print("模式1: 只有空间坐标输入")
    with torch.no_grad():
        output1 = model(x, fx=None, T=T if config['Time_Input'] else None)
        print(f"  输出形状: {output1.shape}")
    
    # 模式2: 有空间坐标和函数值
    print("模式2: 空间坐标 + 函数值输入")
    with torch.no_grad():
        output2 = model(x, fx, T=T if config['Time_Input'] else None)
        print(f"  输出形状: {output2.shape}")
    
    # 模式3: 不使用时间输入
    print("模式3: 不使用时间输入")
    model_no_time = Model(Time_Input=False, **{k:v for k,v in config.items() if k != 'Time_Input'})
    with torch.no_grad():
        output3 = model_no_time(x, fx)
        print(f"  输出形状: {output3.shape}")
    
    print()
    
    # 5. 模型训练示例
    print("5. 模型训练示例...")
    model.train()  # 设置为训练模式
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 生成目标数据
    target = torch.randn_like(output2)
    
    # 前向传播
    pred = model(x, fx, T=T if config['Time_Input'] else None)
    loss = criterion(pred, target)
    
    print(f"训练损失: {loss.item():.6f}")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("反向传播完成")
    print()
    
    # 6. 模型信息
    print("6. 模型详细信息...")
    print(f"模型名称: {model.__name__}")
    print(f"模型结构:")
    print(model)
    
    # 计算模型大小
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数数量: {trainable_params:,}")
    
    # 计算模型大小（MB）
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"  模型大小: {size_all_mb:.2f} MB")

if __name__ == "__main__":
    main() 