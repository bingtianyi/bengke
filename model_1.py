import torch
import os
import torch.nn as nn
import timm
from config import (
    DEVICE, NUM_CLASSES, VIS_BACKBONE_NAME, IR_BACKBONE_NAME,
    VIS_PRETRAINED_PATH, IR_PRETRAINED_PATH, USE_VIS_ONLY, USE_IR_ONLY, DROP_RATE
)

# ===================== 拼接融合模块（用于满足 main.py 的 fusion 属性） =====================
class ConcatFusion(nn.Module):
    def __init__(self, vis_dim, ir_dim):
        super().__init__()
        self.vis_dim = vis_dim
        self.ir_dim = ir_dim

    def forward(self, vis_feat, ir_feat):
        return torch.cat([vis_feat, ir_feat], dim=1)

# ===================== 简单拼接融合模型 =====================
class DualBackboneDualStream(nn.Module):
    """
    双主干 + 简单特征拼接 + 分类头
    - 不包含自适应权重、CFIM/DFIM、自注意力
    - 仅将 VIS 特征和 IR 特征沿通道维拼接，送入 MLP 分类
    """
    def __init__(self):
        super().__init__()
        # 加载双主干（不包含分类头）
        self.vis_backbone = timm.create_model(VIS_BACKBONE_NAME, pretrained=False, num_classes=0)
        self.ir_backbone = timm.create_model(IR_BACKBONE_NAME, pretrained=False, num_classes=0)
        self._load_pretrained_weights(self.vis_backbone, VIS_PRETRAINED_PATH, "VIS")
        self._load_pretrained_weights(self.ir_backbone, IR_PRETRAINED_PATH, "IR")

        # 红外输入适配：1通道 → 3通道
        self.ir_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1)

        # 特征维度（ResNet50: 2048，ConvNeXtV2-Tiny: 768）
        self.vis_feat_dim = 2048
        self.ir_feat_dim = 768
        concat_dim = self.vis_feat_dim + self.ir_feat_dim  # 2816

        # ------------------- 新增：fusion 模块（仅拼接，但提供参数组） -------------------
        self.fusion = ConcatFusion(self.vis_feat_dim, self.ir_feat_dim)

        # 单模态备用分类头（与原模型一致）
        self.vis_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(self.vis_feat_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(1024, NUM_CLASSES)
        )
        self.ir_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(self.ir_feat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(512, NUM_CLASSES)
        )

        # 融合分类头：接收拼接后的特征 (2816维)
        self.fusion_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(concat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(512, NUM_CLASSES)
        )

        # 单模态模式检查
        if USE_VIS_ONLY and USE_IR_ONLY:
            raise ValueError("不能同时开启 USE_VIS_ONLY 和 USE_IR_ONLY")

        # 部分冻结主干（仅微调最后阶段）
        self._freeze_backbones_partially()

    def _freeze_backbones_partially(self):
        """解冻 ResNet50 的 layer4，ConvNeXtV2 的 stages.3；其余冻结"""
        for name, param in self.vis_backbone.named_parameters():
            if 'layer4' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in self.ir_backbone.named_parameters():
            if 'stages.3' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("✅ 解冻主干最后两层 (ResNet layer4, ConvNeXt stages[3])，其余冻结")

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
                raise ValueError("USE_VIS_ONLY=True 时必须传入 vis_x")
            feat = self.vis_backbone(vis_x)
            logits = self.vis_head(feat)
            return logits, None   # 保持元组返回格式

        elif USE_IR_ONLY:
            if ir_x is None:
                raise ValueError("USE_IR_ONLY=True 时必须传入 ir_x")
            ir_x = self.ir_adapter(ir_x)
            feat = self.ir_backbone(ir_x)
            logits = self.ir_head(feat)
            return logits, None

        else:
            # 正常双模态融合模式
            vis_feat = self.vis_backbone(vis_x)                # (B, 2048)
            ir_x_adapted = self.ir_adapter(ir_x)               # (B, 3, H, W)
            ir_feat = self.ir_backbone(ir_x_adapted)           # (B, 768)

            # # 简单拼接融合（核心修改）
            # fused_feat = torch.cat([vis_feat, ir_feat], dim=1) # (B, 2816)
            # logits = self.fusion_head

            # 通过 fusion 模块拼接（现在有 self.fusion 了）
            fused_feat = self.fusion(vis_feat, ir_feat)  # (B, 2816)
            logits = self.fusion_head(fused_feat)

            # 返回元组，与原始接口兼容（训练脚本会解包）
            return logits, fused_feat