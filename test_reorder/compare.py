#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
比较原始CogVideoXBlock与修改后的test_CogVideoXBlock
"""

import torch
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.models.transformers.test_cogvideox_transformer_3d import CogVideoXBlock as test_CogVideoXBlock
from diffusers.models.normalization import CogVideoXLayerNormZero
from diffusers.models.test_normalization import ReorderCogVideoXLayerNormZero
import numpy as np
import matplotlib.pyplot as plt

def compare_blocks(
    batch_size=2,
    num_frames=22,
    height=96,
    width=170,
    channels=32,
    dim=1024,
    num_attention_heads=16,
    attention_head_dim=64,
    time_embed_dim=512,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float16
):
    """比较原始CogVideoXBlock和test_CogVideoXBlock"""
    
    print(f"使用设备: {device}")
    print(f"使用精度: {dtype}")
    
    # 配置参数
    params = {
        "dim": dim,
        "num_attention_heads": num_attention_heads,
        "attention_head_dim": attention_head_dim,
        "time_embed_dim": time_embed_dim,
        "norm_elementwise_affine": True,
        "norm_eps": 1e-5,
    }
    
    # 创建原始模型
    print("\n初始化原始CogVideoXBlock...")
    original_block = CogVideoXBlock(**params).to(device).to(dtype)
    
    # 创建测试模型
    print("初始化测试test_CogVideoXBlock...")
    test_block = test_CogVideoXBlock(**params).to(device).to(dtype)
    
    # 确保两个模型具有相同的权重
    print("\n复制原始模型的权重到测试模型...")
    test_block.load_state_dict(original_block.state_dict(), strict=False)
    
    # 打印关键区别
    print("\n检查关键区别:")
    print(f"原始norm1类型: {type(original_block.norm1).__name__}")
    print(f"测试norm1类型: {type(test_block.norm1).__name__}")
    
    # 准备测试输入
    print("\n生成测试输入...")
    seq_len = 226  # 文本序列长度
    embed_dim = 4096  # 嵌入维度
    T = num_frames//2 * height // 2 * width // 2  # 总时间步
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建相同的输入
    hidden_states = torch.randn(batch_size, T, dim, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    temb = torch.randn(batch_size, time_embed_dim, device=device, dtype=dtype)
    
    # 创建旋转位置嵌入
    freqs_cos = torch.randn(T, attention_head_dim, device=device, dtype=dtype)
    freqs_sin = torch.randn(T, attention_head_dim, device=device, dtype=dtype)
    image_rotary_emb = (freqs_cos, freqs_sin)
    
    # 打印输入形状
    print(f"hidden_states 形状: {hidden_states.shape}")
    print(f"encoder_hidden_states 形状: {encoder_hidden_states.shape}")
    print(f"temb 形状: {temb.shape}")
    print(f"image_rotary_emb[0] 形状: {image_rotary_emb[0].shape}")
    print(f"image_rotary_emb[1] 形状: {image_rotary_emb[1].shape}")
    
    # 关闭梯度计算以加速处理
    with torch.no_grad():
        print("\n运行原始CogVideoXBlock...")
        original_output_hidden, original_output_encoder = original_block(
            hidden_states=hidden_states.clone(),
            encoder_hidden_states=encoder_hidden_states.clone(),
            temb=temb.clone(),
            image_rotary_emb=image_rotary_emb,
        )
        
        print("运行测试test_CogVideoXBlock...")
        # 我们需要找出正确的H,W,F值用于测试模型
        # 通常来说，我们需要动态计算这些值而非硬编码
        # T = H * W * F, H应该是height//2, W应该是width//2, F应该是num_frames
        H = height // 2
        W = width // 2
        F = num_frames//2
        
        # 因为当前test_CogVideoXBlock中硬编码了H=48,W=85,F=11，我们需要修改它
        # 但在测试代码中，我们不能直接修改类，所以需要创建一个新的实例并手动修改其参数
        # 这里我们通过假设计算方式来模拟运行，或者你需要根据实际情况调整
        
        # 打印关键尺寸
        print(f"\n尺寸检查:")
        print(f"T = {T}, H*W*F = {H}*{W}*{F} = {H*W*F}")
        
        # 修改test_block中的norm1和norm2中的尺寸参数
        if hasattr(test_block.norm1, 'Height'):
            print(f"调整norm1的尺寸参数: {test_block.norm1.Height},{test_block.norm1.Width},{test_block.norm1.F} -> {H},{W},{F}")
            test_block.norm1.Height = H
            test_block.norm1.Width = W
            test_block.norm1.F = F
        
        if hasattr(test_block.norm2, 'Height'):
            print(f"调整norm2的尺寸参数: {test_block.norm2.Height},{test_block.norm2.Width},{test_block.norm2.F} -> {H},{W},{F}")
            test_block.norm2.Height = H
            test_block.norm2.Width = W
            test_block.norm2.F = F
        
        test_output_hidden, test_output_encoder = test_block(
            hidden_states=hidden_states.clone(),
            encoder_hidden_states=encoder_hidden_states.clone(),
            temb=temb.clone(),
            image_rotary_emb=image_rotary_emb,
        )
    
    # 比较输出
    print("\n比较输出结果...")
    # 将张量转换为float32进行精确比较
    original_output_hidden_float = original_output_hidden.float()
    test_output_hidden_float = test_output_hidden.float()
    original_output_encoder_float = original_output_encoder.float()
    test_output_encoder_float = test_output_encoder.float()
    
    # 检查hidden_states输出
    hidden_close = torch.allclose(
        original_output_hidden_float, 
        test_output_hidden_float, 
        rtol=1e-3, 
        atol=1e-3
    )
    hidden_max_diff = torch.max(torch.abs(original_output_hidden_float - test_output_hidden_float)).item()
    hidden_mean_diff = torch.mean(torch.abs(original_output_hidden_float - test_output_hidden_float)).item()
    
    # 检查encoder_hidden_states输出
    encoder_close = torch.allclose(
        original_output_encoder_float, 
        test_output_encoder_float, 
        rtol=1e-3, 
        atol=1e-3
    )
    encoder_max_diff = torch.max(torch.abs(original_output_encoder_float - test_output_encoder_float)).item()
    encoder_mean_diff = torch.mean(torch.abs(original_output_encoder_float - test_output_encoder_float)).item()
    
    # 输出比较结果
    print("\nhidden_states输出比较:")
    print(f"- 形状是否相同: {original_output_hidden.shape == test_output_hidden.shape}")
    print(f"- 形状: {original_output_hidden.shape}")
    print(f"- 是否足够接近 (阈值=1e-3): {hidden_close}")
    print(f"- 最大绝对差异: {hidden_max_diff}")
    print(f"- 平均绝对差异: {hidden_mean_diff}")
    
    print("\nencoder_hidden_states输出比较:")
    print(f"- 形状是否相同: {original_output_encoder.shape == test_output_encoder.shape}")
    print(f"- 形状: {original_output_encoder.shape}")
    print(f"- 是否足够接近 (阈值=1e-3): {encoder_close}")
    print(f"- 最大绝对差异: {encoder_max_diff}")
    print(f"- 平均绝对差异: {encoder_mean_diff}")
    
    # 如果差异大，可视化差异
    if not hidden_close:
        # 可视化第一个batch的差异
        diff = torch.abs(original_output_hidden_float - test_output_hidden_float)[0]
        diff = diff.mean(dim=-1)  # 沿通道维度取平均
        
        # 将一维差异数组重塑为三维 [F, H, W]
        F, H, W = 11, 48, 85
        diff_3d = diff.reshape(F, H, W).cpu().numpy()
        
        # 创建多个子图显示差异
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Hidden States 差异热图 (选择帧)', fontsize=16)
        
        # 选择6个帧进行可视化
        frame_indices = [0, 2, 4, 6, 8, 10] if F >= 11 else list(range(min(6, F)))
        vmax = np.max(diff_3d)  # 统一色彩范围
        
        for i, ax in enumerate(axes.flat):
            if i < len(frame_indices):
                frame_idx = frame_indices[i]
                im = ax.imshow(diff_3d[frame_idx], cmap='hot', vmax=vmax)
                ax.set_title(f'帧 {frame_idx}')
                ax.axis('off')
            else:
                ax.axis('off')
        
        # 添加颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='绝对差异')
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig('hidden_diff_frames.png')
        
        # 创建全帧平均差异图
        plt.figure(figsize=(10, 6))
        mean_diff = diff_3d.mean(axis=0)  # 沿帧维度取平均
        plt.imshow(mean_diff, cmap='hot')
        plt.colorbar(label='平均绝对差异')
        plt.title('Hidden States 所有帧平均差异')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('hidden_diff_mean.png')
        
        # 创建最大差异图
        plt.figure(figsize=(10, 6))
        max_diff = diff_3d.max(axis=0)  # 沿帧维度取最大值
        plt.imshow(max_diff, cmap='hot')
        plt.colorbar(label='最大绝对差异')
        plt.title('Hidden States 所有帧最大差异')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('hidden_diff_max.png')
        
        plt.close('all')
        print("已保存差异热图到 hidden_diff_frames.png, hidden_diff_mean.png 和 hidden_diff_max.png")
    
    # 总结
    print("\n总结:")
    if hidden_close and encoder_close:
        print("✓ 测试通过: 两个模块的输出足够接近。")
    else:
        print("✗ 测试失败: 两个模块的输出有显著差异。")
        
    return {
        "hidden_close": hidden_close,
        "encoder_close": encoder_close,
        "hidden_max_diff": hidden_max_diff,
        "hidden_mean_diff": hidden_mean_diff,
        "encoder_max_diff": encoder_max_diff,
        "encoder_mean_diff": encoder_mean_diff
    }

if __name__ == "__main__":
    # 运行测试
    results = compare_blocks(
        batch_size=2,
        num_frames=22,
        height=96,
        width=170,
        channels=32,
        dim=1024,
        num_attention_heads=16,
        attention_head_dim=64,
        time_embed_dim=512,
    )