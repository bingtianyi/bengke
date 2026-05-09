import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    DEVICE, NUM_CLASSES, USE_VIS_ONLY, USE_IR_ONLY, DROP_RATE
)

# -------------------- 轻量级 CNN 主干（无 inplace 操作） --------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=False),          # ❗ 禁用 inplace
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=False)
        )
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity          # ✅ 非原地相加，创建新张量
        out = self.relu(out)
        return out

class LightBackbone(nn.Module):
    def __init__(self, in_channels, feat_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)      # ❗ 禁用 inplace
        )
        self.stage1 = self._make_layer(32, 64, 2, stride=2)
        self.stage2 = self._make_layer(64, 128, 2, stride=2)
        self.stage3 = self._make_layer(128, 256, 2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, feat_dim)

    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [ConvBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(ConvBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# -------------------- 原融合模块（保持不变，已无 inplace 冲突） --------------------
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

class CrossModalAttention(nn.Module):
    def __init__(self, vis_feat_dim=256, ir_feat_dim=256, proj_dim=128, dropout=DROP_RATE):
        super().__init__()
        self.vis_proj = nn.Linear(vis_feat_dim, proj_dim) if vis_feat_dim != proj_dim else nn.Identity()
        self.ir_proj = nn.Linear(ir_feat_dim, proj_dim) if ir_feat_dim != proj_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.cfim = CommonFeatureInjection(proj_dim, dropout)
        self.dfim = DifferentFeatureInjection(proj_dim, dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=4, dropout=dropout, batch_first=True)
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
        fused_feat = self.fusion_mlp(concat_feat)

        fused_feat = fused_feat.unsqueeze(1)
        attn_out, _ = self.self_attn(fused_feat, fused_feat, fused_feat)
        fused_feat = attn_out.squeeze(1)
        return fused_feat

# -------------------- 主模型 --------------------
class DualBackboneDualStream(nn.Module):
    def __init__(self, vis_in=3, ir_in=1, feat_dim=256, proj_dim=128):
        super().__init__()
        self.vis_backbone = LightBackbone(vis_in, feat_dim)
        self.ir_backbone = LightBackbone(ir_in, feat_dim)

        self.vis_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(feat_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=False),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(128, NUM_CLASSES)
        )
        self.ir_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(feat_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=False),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(128, NUM_CLASSES)
        )

        self.fusion = CrossModalAttention(
            vis_feat_dim=feat_dim,
            ir_feat_dim=feat_dim,
            proj_dim=proj_dim,
            dropout=DROP_RATE
        )

        self.fusion_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(proj_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(64, NUM_CLASSES)
        )

        if USE_VIS_ONLY and USE_IR_ONLY:
            raise ValueError("不能同时开启 USE_VIS_ONLY 和 USE_IR_ONLY")

        print("✅ 轻量双主干模型初始化完成（无 inplace 操作），全部参数可训练")

    def forward(self, vis_x=None, ir_x=None):
        if USE_VIS_ONLY:
            if vis_x is None:
                raise ValueError("USE_VIS_ONLY=True时，必须传入vis_x！")
            feat = self.vis_backbone(vis_x)
            out = self.vis_head(feat)
            return out, None
        elif USE_IR_ONLY:
            if ir_x is None:
                raise ValueError("USE_IR_ONLY=True时，必须传入ir_x！")
            feat = self.ir_backbone(ir_x)
            out = self.ir_head(feat)
            return out, None
        else:
            vis_feat = self.vis_backbone(vis_x)
            ir_feat = self.ir_backbone(ir_x)
            fused_feat = self.fusion(vis_feat, ir_feat)
            out = self.fusion_head(fused_feat)
            return out, fused_feat