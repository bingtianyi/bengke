import torch
import os
import torch.nn as nn
import timm
from config import (
    DEVICE, NUM_CLASSES, VIS_BACKBONE_NAME, IR_BACKBONE_NAME,
    VIS_PRETRAINED_PATH, IR_PRETRAINED_PATH, USE_VIS_ONLY, USE_IR_ONLY, DROP_RATE
)

# ===================== SIFusion 消融融合模块（满足 main.py 的 fusion 属性） =====================
class SIFusionFusion(nn.Module):
    def __init__(self, vis_dim=2048, ir_dim=768, embed_dim=256, dropout=DROP_RATE):
        super().__init__()
        self.embed_dim = embed_dim
        self.vis_proj = nn.Linear(vis_dim, embed_dim)
        self.ir_proj = nn.Linear(ir_dim, embed_dim)

        # 光照分类器：从 VIS 投影特征预测光照概率（3类）
        self.illum_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=False),
            nn.Linear(embed_dim // 2, 3),
        )
        # 光照概率到模态权重的映射层
        self.illum_to_modal = nn.Linear(3, 2)

        # 基础门控：基于拼接特征预测权重
        self.base_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, vis_feat_raw, ir_feat_raw):
        vis_proj = self.vis_proj(vis_feat_raw)
        ir_proj = self.ir_proj(ir_feat_raw)

        # 光照感知权重
        illum_logits = self.illum_classifier(vis_proj)
        illum_probs = torch.softmax(illum_logits, dim=1)
        modal_weights_illum = self.illum_to_modal(illum_probs)
        modal_weights_illum = torch.softmax(modal_weights_illum, dim=1)

        # 基础门控权重
        concat_feat = torch.cat([vis_proj, ir_proj], dim=1)
        modal_weights_base = self.base_gate(concat_feat)

        # 平均融合权重
        final_weights = (modal_weights_illum + modal_weights_base) / 2.0
        w_vis = final_weights[:, 0:1]
        w_ir  = final_weights[:, 1:2]

        fused = w_vis * vis_proj + w_ir * ir_proj
        return fused


# ===================== SIFusion 消融融合模型 =====================
class DualBackboneDualStream(nn.Module):
    """
    双主干 + 投影对齐 + 光照感知门控融合 + 分类头
    思想源自 SIFusion 的消融设计：
    - 从 VIS 特征预测光照概率（3类：正常/弱光/黑暗）
    - 光照概率映射为模态权重 (vis_w, ir_w)
    - 基础门控基于拼接特征预测另一组权重
    - 最终权重 = (基础门控 + 光照映射) / 2
    - 加权融合投影特征后分类
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        # 双主干
        self.vis_backbone = timm.create_model(VIS_BACKBONE_NAME, pretrained=False, num_classes=0)
        self.ir_backbone = timm.create_model(IR_BACKBONE_NAME, pretrained=False, num_classes=0)
        self._load_pretrained_weights(self.vis_backbone, VIS_PRETRAINED_PATH, "VIS")
        self._load_pretrained_weights(self.ir_backbone, IR_PRETRAINED_PATH, "IR")

        # 红外输入适配：1通道 → 3通道
        self.ir_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1)

        # 特征维度
        vis_dim = 2048      # ResNet50
        ir_dim = 768        # ConvNeXtV2-Tiny

        # ------------------- fusion 模块（SIFusion 消融） -------------------
        self.fusion = SIFusionFusion(vis_dim, ir_dim, embed_dim, dropout=DROP_RATE)

        # 投影层：将 VIS 和 IR 特征映射到统一维度
        self.vis_proj = nn.Linear(vis_dim, embed_dim)
        self.ir_proj = nn.Linear(ir_dim, embed_dim)

        # 光照分类器：从 VIS 投影特征预测光照概率（3类）
        self.illum_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 3),   # 3 类光照：正常/弱光/黑暗
        )

        # 光照概率到模态权重的映射层
        self.illum_to_modal = nn.Linear(3, 2)   # 输出 2 个权重（未归一化）

        # 基础门控：基于拼接特征预测权重
        self.base_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 2),
            nn.Softmax(dim=-1)
        )

        # 单模态备用分类头（与原模型一致）
        self.vis_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(vis_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(1024, NUM_CLASSES)
        )
        self.ir_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(ir_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(512, NUM_CLASSES)
        )

        # 融合分类头
        self.fusion_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(embed_dim // 2, NUM_CLASSES)
        )

        if USE_VIS_ONLY and USE_IR_ONLY:
            raise ValueError("不能同时开启 USE_VIS_ONLY 和 USE_IR_ONLY")

        self._freeze_backbones_partially()

    def _freeze_backbones_partially(self):
        """解冻主干最后阶段，其余冻结"""
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
            # 1. 提取原始特征
            vis_feat_raw = self.vis_backbone(vis_x)                 # (B, 2048)
            ir_x_adapted = self.ir_adapter(ir_x)
            ir_feat_raw = self.ir_backbone(ir_x_adapted)            # (B, 768)

            fused_feat = self.fusion(vis_feat_raw, ir_feat_raw)

            # # 2. 投影到统一维度
            # vis_proj = self.vis_proj(vis_feat_raw)                  # (B, embed_dim)
            # ir_proj = self.ir_proj(ir_feat_raw)                     # (B, embed_dim)
            #
            # # 3. 光照感知权重
            # #   从 VIS 投影特征预测光照类别概率（3类）
            # illum_logits = self.illum_classifier(vis_proj)          # (B, 3)
            # illum_probs = torch.softmax(illum_logits, dim=1)        # (B, 3)
            # #   映射为模态权重
            # modal_weights_illum = self.illum_to_modal(illum_probs)  # (B, 2)
            # modal_weights_illum = torch.softmax(modal_weights_illum, dim=1)  # 归一化为概率分布
            #
            # # 4. 基础特征门控权重
            # concat_feat = torch.cat([vis_proj, ir_proj], dim=1)     # (B, embed_dim*2)
            # modal_weights_base = self.base_gate(concat_feat)        # (B, 2)，已 softmax
            #
            # # 5. 融合权重：取平均
            # final_weights = (modal_weights_illum + modal_weights_base) / 2.0   # (B, 2)
            # w_vis = final_weights[:, 0:1]   # (B, 1)
            # w_ir  = final_weights[:, 1:2]   # (B, 1)
            #
            # # 6. 加权融合特征
            # fused_feat = w_vis * vis_proj + w_ir * ir_proj          # (B, embed_dim)

            # 7. 分类
            logits = self.fusion_head(fused_feat)

            return logits, fused_feat