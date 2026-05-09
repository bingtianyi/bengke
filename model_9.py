# model_9.py
# 基于 model.py，将可见光与红外主干网络均改为 ResNet-50（2048维特征），并调整所有对应维度

import torch
import os
import torch.nn as nn
import timm
from config import (
    DEVICE, NUM_CLASSES,
    VIS_PRETRAINED_PATH, IR_PRETRAINED_PATH,
    USE_VIS_ONLY, USE_IR_ONLY, DROP_RATE
)

# ===================== CFIM / DFIM =====================
class CommonFeatureInjection(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b):
        common = (a + b) / 2.0
        common_transformed = self.fc(self.norm(common))
        common_enhanced = common + self.dropout(common_transformed)
        return a + common_enhanced, b + common_enhanced

class DifferentFeatureInjection(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b):
        diff = a - b
        diff_transformed = self.fc(diff)
        diff_enhanced = diff + self.dropout(diff_transformed)
        return a + diff_enhanced, b - diff_enhanced

# ===================== 跨模态融合（带自注意力增强） =====================
class CrossModalAttention(nn.Module):
    def __init__(self, vis_feat_dim=2048, ir_feat_dim=2048, proj_dim=128, dropout=DROP_RATE):
        super().__init__()
        self.vis_proj = nn.Linear(vis_feat_dim, proj_dim) if vis_feat_dim != proj_dim else nn.Identity()
        self.ir_proj = nn.Linear(ir_feat_dim, proj_dim) if ir_feat_dim != proj_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.cfim = CommonFeatureInjection(proj_dim, dropout)
        self.dfim = DifferentFeatureInjection(proj_dim, dropout)

        # 自注意力增强（可选，提升关键区域表达）
        self.self_attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=4, dropout=0.1, batch_first=True)

        # 融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim * 4, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, vis_feat, ir_feat):
        vis_feat = self.vis_proj(vis_feat)
        ir_feat = self.ir_proj(ir_feat)
        vis_feat = self.dropout(vis_feat)
        ir_feat = self.dropout(ir_feat)

        vis_cf, ir_cf = self.cfim(vis_feat, ir_feat)
        vis_df, ir_df = self.dfim(vis_feat, ir_feat)

        concat_feat = torch.cat([vis_cf, ir_cf, vis_df, ir_df], dim=1)
        fused_feat = self.fusion_mlp(concat_feat)          # (B, proj_dim)

        # 自注意力增强
        fused_feat = fused_feat.unsqueeze(1)               # (B, 1, proj_dim)
        attn_out, _ = self.self_attn(fused_feat, fused_feat, fused_feat)
        fused_feat = attn_out.squeeze(1)                   # (B, proj_dim)

        return fused_feat

# ===================== 双主干双流模型 =====================
class DualBackboneDualStream(nn.Module):
    def __init__(self):
        super().__init__()
        # 可见光主干：ResNet-50，输出 2048 维
        self.vis_backbone = timm.create_model('resnet50', pretrained=False, num_classes=0)
        # 红外主干：同样使用 ResNet-50，输出 2048 维
        self.ir_backbone = timm.create_model('resnet50', pretrained=False, num_classes=0)

        # 加载预训练权重（路径从 config 读取）
        self._load_pretrained_weights(self.vis_backbone, VIS_PRETRAINED_PATH, "VIS")
        self._load_pretrained_weights(self.ir_backbone, IR_PRETRAINED_PATH, "IR")

        self.ir_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1)

        # 单模态分类头（备用），特征维度均为 2048
        self.vis_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(1024, NUM_CLASSES)
        )
        self.ir_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(1024, NUM_CLASSES)
        )

        # 融合模块：vis 与 ir 特征维度均为 2048
        self.fusion = CrossModalAttention(vis_feat_dim=2048, ir_feat_dim=2048, proj_dim=128, dropout=DROP_RATE)

        # 融合分类头
        self.fusion_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(64, NUM_CLASSES)
        )

        if USE_VIS_ONLY and USE_IR_ONLY:
            raise ValueError("不能同时开启 USE_VIS_ONLY 和 USE_IR_ONLY")

        # 部分冻结主干：解冻最后两层，其余冻结
        self._freeze_backbones_partially()

    def _freeze_backbones_partially(self):
        """解冻 ResNet50 的 layer3, layer4；其余冻结"""
        for name, param in self.vis_backbone.named_parameters():
            if 'layer3' in name or 'layer4' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in self.ir_backbone.named_parameters():
            if 'layer3' in name or 'layer4' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("✅ 解冻主干最后两层 (ResNet50 layer3,4)，其余冻结")

    def _load_pretrained_weights(self, backbone, weight_path, modal):
        if not weight_path or not os.path.exists(weight_path):
            print(f"⚠️ {modal}分支权重缺失：{weight_path}，从头训练")
            return
        try:
            pretrained_dict = torch.load(weight_path, map_location=DEVICE)
            if 'model' in pretrained_dict:
                pretrained_dict = pretrained_dict['model']
            if 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']
            filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('head.')}
            backbone.load_state_dict(filtered_dict, strict=False)
            print(f"✅ {modal}分支权重加载成功：{weight_path}")
        except Exception as e:
            print(f"❌ {modal}分支权重加载失败：{e}，从头训练")

    def forward(self, vis_x=None, ir_x=None):
        if USE_VIS_ONLY:
            if vis_x is None:
                raise ValueError("USE_VIS_ONLY=True时，必须传入vis_x！")
            feat = self.vis_backbone(vis_x)
            out = self.vis_head(feat)
            return out, None   # 兼容返回格式
        elif USE_IR_ONLY:
            if ir_x is None:
                raise ValueError("USE_IR_ONLY=True时，必须传入ir_x！")
            ir_x = self.ir_adapter(ir_x)
            feat = self.ir_backbone(ir_x)
            out = self.ir_head(feat)
            return out, None
        else:
            vis_feat = self.vis_backbone(vis_x)
            ir_x = self.ir_adapter(ir_x)
            ir_feat = self.ir_backbone(ir_x)
            fused_feat = self.fusion(vis_feat, ir_feat)
            out = self.fusion_head(fused_feat)
            return out, fused_feat
