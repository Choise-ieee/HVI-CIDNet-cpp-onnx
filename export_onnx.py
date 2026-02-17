#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_onnx_dynamic.py - 导出支持动态alpha参数的ONNX模型
输入: input (图像), alpha_s, alpha_i, gamma
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import safetensors.torch as sf
from huggingface_hub import hf_hub_download
import argparse


# ============ 注册自定义atan2算子 ============
def register_atan2_symbolic():
    from torch.onnx import register_custom_op_symbolic
    
    def atan2_symbolic(g, y, x):
        x_sq = g.op("Mul", x, x)
        y_sq = g.op("Mul", y, y)
        sum_sq = g.op("Add", x_sq, y_sq)
        sqrt_val = g.op("Sqrt", sum_sq)
        denom = g.op("Add", sqrt_val, x)
        ratio = g.op("Div", y, denom)
        atan_val = g.op("Atan", ratio)
        two = g.op("Constant", value_t=torch.tensor([2.0]))
        return g.op("Mul", two, atan_val)
    
    register_custom_op_symbolic('aten::atan2', atan2_symbolic, 11)
    print("✓ Registered custom atan2 symbolic function")


register_atan2_symbolic()


# ============ 替换torch.atan2 ============
def safe_atan2(y, x, eps=1e-7):
    denom = torch.sqrt(x * x + y * y) + x
    safe_denom = torch.where(torch.abs(denom) < eps,
                             torch.sign(denom) * eps, denom)
    return 2.0 * torch.atan(y / safe_denom)


_original_atan2 = torch.atan2
torch.atan2 = safe_atan2
print("✓ Replaced torch.atan2 with safe_atan2")


# ============ 导入模型 ============
from net.CIDNet import CIDNet


class CIDNetDynamicAlpha(nn.Module):
    """
    包装CIDNet，将alpha_s、alpha_i、gamma作为输入参数
    """
    def __init__(self, cidnet_model):
        super(CIDNetDynamicAlpha, self).__init__()
        self.model = cidnet_model
        
    def forward(self, x, alpha_s, alpha_i, gamma):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            alpha_s: 标量 [B] 或 [B, 1, 1, 1]
            alpha_i: 标量 [B] 或 [B, 1, 1, 1]
            gamma: 标量 [B] 或 [B, 1, 1, 1]
        """
        # 应用gamma校正
        x = x ** gamma
        
        # 设置alpha参数
        self.model.trans.alpha_s = alpha_s
        self.model.trans.alpha = alpha_i
        self.model.trans.gated = True
        self.model.trans.gated2 = True
        
        # 前向推理
        output = self.model(x)
        
        return output


def from_pretrained(cls, pretrained_model_name_or_path: str):
    model_id = str(pretrained_model_name_or_path)
    
    try:
        config_file = hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model")
    except:
        pass
    
    model_file = hf_hub_download(repo_id=model_id, filename="model.safetensors", repo_type="model")
    state_dict = sf.load_file(model_file)
    cls.load_state_dict(state_dict, strict=False)
    return cls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export HVI-CIDNet to ONNX with dynamic parameters')
    parser.add_argument('--path', type=str, default="Fediory/HVI-CIDNet-LOLv1-wperc")
    parser.add_argument('--output', type=str, default="hvi_cidnet_dynamic.onnx")
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("HVI-CIDNet ONNX Export (Dynamic Parameters)")
    print("="*60)
    print(f"Export size: {args.height}x{args.width}")
    print(f"Inputs: input, alpha_s, alpha_i, gamma")
    
    # 加载模型
    print("\n[1/3] Loading model...")
    base_model = CIDNet().cuda()
    base_model = from_pretrained(cls=base_model, pretrained_model_name_or_path=args.path)
    base_model.eval()
    
    # 包装模型
    model = CIDNetDynamicAlpha(base_model)
    model.eval()
    
    # 创建测试输入
    print(f"\n[2/3] Creating test inputs ({args.height}x{args.width})...")
    dummy_input = torch.randn(1, 3, args.height, args.width).cuda()
    dummy_alpha_s = torch.tensor([1.0]).cuda()
    dummy_alpha_i = torch.tensor([1.0]).cuda()
    dummy_gamma = torch.tensor([1.0]).cuda()
    
    # 导出ONNX
    print(f"\n[3/3] Exporting to ONNX (opset: {args.opset})...")
    
    try:
        torch.onnx.export(
            model,
            (dummy_input, dummy_alpha_s, dummy_alpha_i, dummy_gamma),
            args.output,
            input_names=['input', 'alpha_s', 'alpha_i', 'gamma'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'},
                'alpha_s': {0: 'batch_size'},
                'alpha_i': {0: 'batch_size'},
                'gamma': {0: 'batch_size'}
            },
            opset_version=args.opset,
            do_constant_folding=False,  # 关键：避免形状固化
        )
        
        # 验证
        print("\n[Verifying exported model]")
        import onnx
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        
        print("\nModel inputs:")
        for inp in onnx_model.graph.input:
            shape = [d.dim_value if d.dim_value else d.dim_param 
                    for d in inp.type.tensor_type.shape.dim]
            print(f"  - {inp.name}: {shape}")
        
        # 测试推理
        print("\n[Testing with onnxruntime]")
        import onnxruntime as ort
        import numpy as np
        
        sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
        
        # 测试不同尺寸和参数
        test_cases = [
            (args.height, args.width, 1.0, 1.0, 1.0),
            (256, 256, 1.0, 1.0, 1.0),
            (512, 512, 1.2, 0.8, 1.0),
            (320, 480, 1.5, 1.0, 0.9),
        ]
        
        for h, w, a_s, a_i, g in test_cases:
            test_input = np.random.randn(1, 3, h, w).astype(np.float32)
            result = sess.run(None, {
                'input': test_input,
                'alpha_s': np.array([a_s], dtype=np.float32),
                'alpha_i': np.array([a_i], dtype=np.float32),
                'gamma': np.array([g], dtype=np.float32)
            })
            print(f"  {h}x{w}, alpha_s={a_s}, alpha_i={a_i}, gamma={g}: ✓ {result[0].shape}")
        
        print("\n" + "="*60)
        print("✓ SUCCESS!")
        print("="*60)
        print(f"Saved to: {args.output}")
        print(f"Size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")
        print("\nUsage:")
        print("  C++: --alpha_s 1.0 --alpha_i 1.0 --gamma 1.0")
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        torch.atan2 = _original_atan2
