import torch
import os
import torch.nn as nn
import timm
from config import (
    DEVICE, NUM_CLASSES, VIS_BACKBONE_NAME, IR_BACKBONE_NAME,
    VIS_PRETRAINED_PATH, IR_PRETRAINED_PATH, USE_VIS_ONLY, USE_IR_ONLY, DROP_RATE
)

# ===================== 门控融合模块（用于满足 main.py 的 fusion 属性） =====================
class GatedFusion(nn.Module):
    def __init__(self, vis_dim, ir_dim, dropout=DROP_RATE):
        super().__init__()
        self.ir_proj = nn.Linear(ir_dim, vis_dim)   # IR 投影到 VIS 维度
        self.gate = nn.Sequential(
            nn.Linear(vis_dim + ir_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, vis_feat, ir_feat_raw):
        # 投影 IR
        ir_feat = self.ir_proj(ir_feat_raw)
        # 拼接并计算门控
        concat = torch.cat([vis_feat, ir_feat_raw], dim=1)
        weights = self.gate(concat)
        g_vis = weights[:, 0:1]
        g_ir = weights[:, 1:2]
        fused = g_vis * vis_feat + g_ir * ir_feat
        return fused

# ===================== 门控融合模型 =====================
class DualBackboneDualStream(nn.Module):
    """
    双主干 + 门控融合 + 分类头
    - 通过一个可学习的门控网络预测每个模态的权重（和为1）
    - 加权融合后送入分类头
    """
    def __init__(self):
        super().__init__()
        # 双主干
        self.vis_backbone = timm.create_model(VIS_BACKBONE_NAME, pretrained=False, num_classes=0)
        self.ir_backbone = timm.create_model(IR_BACKBONE_NAME, pretrained=False, num_classes=0)
        self._load_pretrained_weights(self.vis_backbone, VIS_PRETRAINED_PATH, "VIS")
        self._load_pretrained_weights(self.ir_backbone, IR_PRETRAINED_PATH, "IR")

        # 红外输入适配：1通道 → 3通道
        self.ir_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1)

        # 特征维度（ResNet50: 2048，ConvNeXtV2-Tiny: 768）
        self.vis_feat_dim = 2048
        self.ir_feat_dim = 768

        # ------------------- 新增：fusion 模块（门控 + 投影） -------------------
        self.fusion = GatedFusion(self.vis_feat_dim, self.ir_feat_dim, dropout=DROP_RATE)

        # concat_dim = self.vis_feat_dim + self.ir_feat_dim  # 2816
        #
        # # 门控网络：输入拼接特征，输出2个权重（VIS权重，IR权重）
        # self.gate = nn.Sequential(
        #     nn.Linear(concat_dim, 128),
        #     nn.ReLU(),
        #     nn.Dropout(DROP_RATE * 0.5),
        #     nn.Linear(128, 2),
        #     nn.Softmax(dim=1)               # 确保 g_vis + g_ir = 1
        # )

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

        # 融合分类头：接收加权融合后的特征（维度与单模态特征相同，仍为2048）
        # 注意：加权求和后特征维度与 VIS 特征一致（我们选择将 IR 特征投影到 VIS 维度）
        self.ir_proj = nn.Linear(self.ir_feat_dim, self.vis_feat_dim)   # 768 → 2048
        self.fusion_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(self.vis_feat_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(1024, NUM_CLASSES)
        )

        if USE_VIS_ONLY and USE_IR_ONLY:
            raise ValueError("不能同时开启 USE_VIS_ONLY 和 USE_IR_ONLY")

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
            return logits, None

        elif USE_IR_ONLY:
            if ir_x is None:
                raise ValueError("USE_IR_ONLY=True 时必须传入 ir_x")
            ir_x = self.ir_adapter(ir_x)
            feat = self.ir_backbone(ir_x)
            logits = self.ir_head(feat)
            return logits, None

        else:
            # 提取特征
            vis_feat = self.vis_backbone(vis_x)                # (B, 2048)
            ir_x_adapted = self.ir_adapter(ir_x)               # (B, 3, H, W)
            ir_feat_raw = self.ir_backbone(ir_x_adapted)       # (B, 768)

            # # 将 IR 特征投影到与 VIS 相同的维度，便于加权求和
            # ir_feat = self.ir_proj(ir_feat_raw)                # (B, 2048)
            #
            # # 门控权重预测：基于拼接特征
            # concat_feat = torch.cat([vis_feat, ir_feat_raw], dim=1)   # (B, 2816)
            # gate_weights = self.gate(concat_feat)                     # (B, 2)
            #
            # # 加权融合
            # g_vis = gate_weights[:, 0:1]   # (B, 1)
            # g_ir  = gate_weights[:, 1:2]   # (B, 1)
            # fused_feat = g_vis * vis_feat + g_ir * ir_feat   # (B, 2048)

            # 使用 fusion 模块进行门控加权融合
            fused_feat = self.fusion(vis_feat, ir_feat_raw)  # (B, 2048)

            # 分类
            logits = self.fusion_head(fused_feat)

            return logits, fused_feat