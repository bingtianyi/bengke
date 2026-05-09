import torch
import os
import torch.nn as nn
import timm
from config import (
    DEVICE, NUM_CLASSES, VIS_BACKBONE_NAME, IR_BACKBONE_NAME,
    VIS_PRETRAINED_PATH, IR_PRETRAINED_PATH, USE_VIS_ONLY, USE_IR_ONLY, DROP_RATE
)

# ===================== STCAN 消融融合模块（用于满足 main.py 的 fusion 属性） =====================
class STCANFusion(nn.Module):
    def __init__(self, vis_dim=2048, ir_dim=768, embed_dim=256, dropout=DROP_RATE):
        super().__init__()
        self.embed_dim = embed_dim
        self.vis_proj = nn.Linear(vis_dim, embed_dim)
        self.ir_proj = nn.Linear(ir_dim, embed_dim)

        self.cfim = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.dfim = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, vis_feat_raw, ir_feat_raw):
        vis_proj = self.vis_proj(vis_feat_raw)
        ir_proj = self.ir_proj(ir_feat_raw)

        concat_proj = torch.cat([vis_proj, ir_proj], dim=1)
        cfim_feat = self.cfim(concat_proj)

        diff = vis_proj - ir_proj
        dfim_feat = self.dfim(diff)

        concat_both = torch.cat([cfim_feat, dfim_feat], dim=1)
        gate_weights = self.gate(concat_both)
        g_cfim = gate_weights[:, 0:1]
        g_dfim = gate_weights[:, 1:2]
        fused = g_cfim * cfim_feat + g_dfim * dfim_feat
        return fused

# ===================== STCAN 消融融合模型 =====================
class DualBackboneDualStream(nn.Module):
    """
    双主干 + 投影对齐 + CFIM-like 共享分支 + DFIM-like 差异分支 + 门控融合 + 分类头
    思想源自 STCAN 的消融设计：
    - CFIM 学习跨模态公共信号
    - DFIM 学习互补差异信号
    - 门控网络动态加权两者
    """
    def __init__(self, embed_dim=256):
        """
        Args:
            embed_dim: 投影后的特征维度（用于融合的统一空间）
        """
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

        # ------------------- fusion 模块（STCAN 消融） -------------------
        self.fusion = STCANFusion(vis_dim, ir_dim, embed_dim, dropout=DROP_RATE)

        # 投影层：将 VIS 和 IR 特征映射到统一维度
        self.vis_proj = nn.Linear(vis_dim, embed_dim)
        self.ir_proj = nn.Linear(ir_dim, embed_dim)

        # CFIM-like 分支：输入拼接特征 [vis_proj, ir_proj]，输出共享特征
        self.cfim = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # DFIM-like 分支：输入差异特征 (vis_proj - ir_proj)，输出互补特征
        self.dfim = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 门控网络：基于拼接的 CFIM 和 DFIM 输出，预测两个权重
        self.st_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 2),
            nn.Softmax(dim=-1)                # 输出 (g_cfim, g_dfim)，和为1
        )

        # 单模态备用分类头（与原模型一致，用于消融）
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
            # # 3. CFIM-like：共享特征注入
            # concat_proj = torch.cat([vis_proj, ir_proj], dim=1)     # (B, embed_dim*2)
            # cfim_feat = self.cfim(concat_proj)                      # (B, embed_dim)
            #
            # # 4. DFIM-like：差异特征注入（使用投影后特征的差异）
            # diff = vis_proj - ir_proj                               # (B, embed_dim)
            # dfim_feat = self.dfim(diff)                             # (B, embed_dim)
            #
            # # 5. 门控融合
            # concat_both = torch.cat([cfim_feat, dfim_feat], dim=1)  # (B, embed_dim*2)
            # gate_weights = self.st_gate(concat_both)                # (B, 2)
            # g_cfim = gate_weights[:, 0:1]
            # g_dfim = gate_weights[:, 1:2]
            # fused_feat = g_cfim * cfim_feat + g_dfim * dfim_feat    # (B, embed_dim)

            # 6. 分类
            logits = self.fusion_head(fused_feat)

            return logits, fused_feat