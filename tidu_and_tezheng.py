#
# #!/usr/bin/env python3
# """
# ablation_analysis.py — 针对 model.py (CFIM+DFIM+SelfAttention) 的消融实验深度分析
# 修复版：解决 self_attn 替换为 Identity 时的参数匹配问题
# """
#
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# import copy
# from PIL import Image
# from torchvision import transforms
#
# # --------------------- 配置 ---------------------
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# IMAGE_SIZE = 224
# MODEL_PATH = 'output/best_model.pth'          # 训练好的模型路径
# #TEST_VIS_PATH = 'data/oulu_aug_test4/train/vis/Strong/anger/P001_013_flip.jpeg'         # 一张测试可见光图像
# #TEST_IR_PATH = 'data/oulu_aug_test4/train/ir/Strong/anger/P001_013_flip.jpeg'          # 一张测试红外图像
# TEST_IR_PATH = 'data/oulu_aug_test4/val/ir/Strong/happiness/P080_020_flip.jpeg'
# TEST_VIS_PATH = 'data/oulu_aug_test4/val/vis/Strong/happiness/P080_020_flip.jpeg'
#
#
# # --------------------- 导入模型 ---------------------
# from model import DualBackboneDualStream
#
# # 加载模型
# model = DualBackboneDualStream().to(DEVICE)
# checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
# model.load_state_dict(checkpoint['model_state_dict'])
# print(f"✅ 模型加载成功，最佳验证精度: {checkpoint.get('best_val_acc', 'N/A')}")
#
# # --------------------- 数据预处理 ---------------------
# def load_image(path, is_ir=False):
#     if is_ir:
#         img = Image.open(path).convert('L')
#         transform = transforms.Compose([
#             transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5], std=[0.5])
#         ])
#     else:
#         img = Image.open(path).convert('RGB')
#         transform = transforms.Compose([
#             transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#     return transform(img).unsqueeze(0).to(DEVICE)
#
# vis_tensor = load_image(TEST_VIS_PATH, is_ir=False)
# ir_tensor = load_image(TEST_IR_PATH, is_ir=True)
#
# # =====================================================================
# # 1️⃣ 梯度贡献分析
# # =====================================================================
# print("="*60)
# print("1. 梯度贡献分析（各模块平均梯度范数）")
# model.train()
#
# outputs, fused_feat = model(vis_tensor, ir_tensor)
# loss = outputs[:, 0].sum()
# model.zero_grad()
# loss.backward()
#
# grad_norms = {}
# def get_module_grad(module, name):
#     grads = [p.grad.norm().item() for p in module.parameters() if p.grad is not None]
#     grad_norms[name] = np.mean(grads) if grads else 0.0
#
# get_module_grad(model.vis_backbone, 'VIS Backbone (ResNet50)')
# get_module_grad(model.ir_backbone, 'IR Backbone (ConvNeXtV2)')
# get_module_grad(model.ir_adapter, 'IR Adapter (1→3 Conv)')
# get_module_grad(model.fusion, 'Fusion (CrossModalAttention)')
# get_module_grad(model.fusion_head, 'Fusion Head')
# get_module_grad(model.vis_head, 'VIS Head (optional)')
# get_module_grad(model.ir_head, 'IR Head (optional)')
#
# plt.figure(figsize=(12, 6))
# modules = list(grad_norms.keys())
# values = list(grad_norms.values())
# bars = plt.bar(modules, values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#95a5a6', '#34495e'])
# plt.ylabel('Average Gradient Norm', fontsize=14)
# plt.title('Gradient Contribution by Module', fontsize=16)
# plt.xticks(rotation=30, ha='right')
# for bar, val in zip(bars, values):
#     if val > 0:
#         plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
#                  f'{val:.2e}', ha='center', va='bottom', fontsize=9)
# plt.tight_layout()
# plt.savefig('gradient_contribution_model.png', dpi=300)
# plt.show()
# print("梯度贡献图已保存\n")
#
# # =====================================================================
# # 2️⃣ 特征可视化：完整模型 vs 去除自注意力
# # =====================================================================
# print("="*60)
# print("2. 特征可视化归因 (完整模型 vs 去除 Self-Attention)")
#
# # 自定义 Identity 模块，兼容 self_attn 的调用方式
# class IdentityMHA(nn.Module):
#     def forward(self, query, key, value, *args, **kwargs):
#         # self_attn 返回 (attn_output, attn_weights)，这里只返回 query 和 None
#         return query, None
#
# model.eval()
#
# def hook_fn(module, input, output):
#     hook_fn.feature = output.detach()
#
# # 注册 hook 到融合 MLP 的输出（自注意力之前）
# target_layer = model.fusion.fusion_mlp
# handle = target_layer.register_forward_hook(hook_fn)
#
# with torch.no_grad():
#     outputs_full, fused_full = model(vis_tensor, ir_tensor)
#     feat_full = hook_fn.feature.squeeze(0).cpu().numpy()
# handle.remove()
#
# # 构建消融模型：用 IdentityMHA 替换 self_attn
# model_ablated = copy.deepcopy(model)
# model_ablated.eval()
# model_ablated.fusion.self_attn = IdentityMHA()
#
# # 获取消融模型的融合 MLP
# target_layer_ab = model_ablated.fusion.fusion_mlp
# handle_ab = target_layer_ab.register_forward_hook(hook_fn)
#
# with torch.no_grad():
#     outputs_ab, fused_ab = model_ablated(vis_tensor, ir_tensor)
#     feat_ab = hook_fn.feature.squeeze(0).cpu().numpy()
# handle_ab.remove()
#
# # 绘制特征对比图（一维向量情况）
# if len(feat_full.shape) == 1:
#     fig, ax = plt.subplots(figsize=(14, 6))
#     x = np.arange(len(feat_full))
#     width = 0.35
#     ax.bar(x - width/2, feat_full, width, label='With Self-Attention', color='#e74c3c', alpha=0.8)
#     ax.bar(x + width/2, feat_ab, width, label='Without Self-Attention', color='#3498db', alpha=0.8)
#     ax.set_xlabel('Feature Dimension', fontsize=14)
#     ax.set_ylabel('Activation Value', fontsize=14)
#     ax.set_title('Fusion MLP Output (Before Self-Attention) Comparison', fontsize=16)
#     ax.legend()
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig('feature_vector_compare_selfattn.png', dpi=300)
#     plt.show()
#     print("特征向量对比图已保存\n")
# else:
#     np.save('feat_full.npy', feat_full)
#     np.save('feat_ab.npy', feat_ab)
#     print("特征维度大于1，已保存为npy文件\n")
#
# print("✅ 消融分析完成。")


#!/usr/bin/env python3
"""
ablation_analysis.py — 深度消融分析：
1. 梯度贡献（去除VIS/IR Head，基于单样本）
2. 特征可视化：完整融合 vs 去除融合模块（拼接+线性投影），基于整个验证集统计
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import OuluCASIADataset      # 请根据实际路径调整导入
from config import DEVICE, BATCH_SIZE, TARGET_DATA_ROOT

# --------------------- 配置 ---------------------
IMAGE_SIZE = 224
MODEL_PATH = 'output/best_model.pth'

# --------------------- 加载完整模型 ---------------------
from model import DualBackboneDualStream
model = DualBackboneDualStream().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ 模型加载成功，最佳验证精度: {checkpoint.get('best_val_acc', 'N/A')}")

# --------------------- 构建消融模型（Concat+Proj 替代融合模块） ---------------------
class ConcatProjectFusion(nn.Module):
    def __init__(self, vis_dim=2048, ir_dim=768, out_dim=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vis_dim + ir_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    def forward(self, vis_feat, ir_feat):
        x = torch.cat([vis_feat, ir_feat], dim=1)
        return self.proj(x)

model_ablation = DualBackboneDualStream().to(DEVICE)
# 加载主干权重，忽略缺失的融合模块（strict=False）
model_ablation.load_state_dict(checkpoint['model_state_dict'], strict=False)
model_ablation.fusion = ConcatProjectFusion().to(DEVICE)
model_ablation.eval()
model.eval()

# =====================================================================
# 1️⃣ 梯度贡献分析（基于单样本，移除 VIS Head、IR Head）
# =====================================================================
print("="*60)
print("1. 梯度贡献分析（各模块平均梯度范数）")

# 使用单张图片计算梯度（也可用验证集平均，但意义相似）
from PIL import Image
from torchvision import transforms
TEST_VIS_PATH = 'data/oulu_aug_test4/val/vis/Strong/happiness/P080_020_flip.jpeg'
TEST_IR_PATH  = 'data/oulu_aug_test4/val/ir/Strong/happiness/P080_020_flip.jpeg'

def load_image(path, is_ir=False):
    if is_ir:
        img = Image.open(path).convert('L')
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        img = Image.open(path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform(img).unsqueeze(0).to(DEVICE)

vis_tensor = load_image(TEST_VIS_PATH, is_ir=False)
ir_tensor = load_image(TEST_IR_PATH, is_ir=True)

model.train()
outputs, _ = model(vis_tensor, ir_tensor)
loss = outputs[:, 0].sum()
model.zero_grad()
loss.backward()

grad_norms = {}
def get_module_grad(module, name):
    grads = [p.grad.norm().item() for p in module.parameters() if p.grad is not None]
    grad_norms[name] = np.mean(grads) if grads else 0.0

get_module_grad(model.vis_backbone, 'VIS Backbone (ResNet50)')
get_module_grad(model.ir_backbone, 'IR Backbone (ConvNeXtV2)')
#get_module_grad(model.ir_adapter, 'IR Adapter (1→3 Conv)')
get_module_grad(model.fusion, 'Fusion (CrossModalAttention)')
#get_module_grad(model.fusion_head, 'Fusion Head')

plt.figure(figsize=(10, 5))
modules = list(grad_norms.keys())
values = list(grad_norms.values())
bars = plt.bar(modules, values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'])
plt.ylabel('Average Gradient Norm', fontsize=14)
plt.title('Gradient Contribution by Module', fontsize=16)
plt.xticks(rotation=20, ha='right')
for bar, val in zip(bars, values):
    if val > 0:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                 f'{val:.2e}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('gradient_contribution_core_modules.png', dpi=300)
plt.close()
print("梯度贡献图已保存\n")

# =====================================================================
# 2️⃣ 特征可视化归因：完整融合模块 vs 去除融合模块（基于整个验证集平均特征）
# =====================================================================
print("="*60)
print("2. 特征可视化对比（验证集平均特征）：完整融合 vs 拼接投影")

# 加载验证集
val_dataset = OuluCASIADataset(split='val', light_scenario=None)  # 不筛选光照
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

all_feat_full = []
all_feat_ab = []

with torch.no_grad():
    for vis, ir, labels in val_loader:
        vis, ir = vis.to(DEVICE), ir.to(DEVICE)
        _, fused_full = model(vis, ir)
        _, fused_ab   = model_ablation(vis, ir)
        all_feat_full.append(fused_full.cpu().numpy())
        all_feat_ab.append(fused_ab.cpu().numpy())

# 拼接所有特征
feat_full_all = np.concatenate(all_feat_full, axis=0)  # shape: (N, 128)
feat_ab_all   = np.concatenate(all_feat_ab, axis=0)    # shape: (N, 128)

# 计算每个维度的平均激活值
mean_full = np.mean(feat_full_all, axis=0)
mean_ab   = np.mean(feat_ab_all, axis=0)

# 绘图：平均特征对比
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(mean_full))
width = 0.35
ax.bar(x - width/2, mean_full, width, label='With Fusion Module', color='#e74c3c', alpha=0.8)
ax.bar(x + width/2, mean_ab, width, label='Without Fusion Module (Concat+Proj)', color='#3498db', alpha=0.8)
ax.set_xlabel('Feature Dimension', fontsize=14)
ax.set_ylabel('Average Activation', fontsize=14)
ax.set_title('Mean Fusion Feature Comparison (Validation Set)', fontsize=16)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('feature_fusion_vs_concat_valavg.png', dpi=300)
plt.close()
print("验证集平均特征对比图已保存\n")

print("✅ 消融分析完成。")