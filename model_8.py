#
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from config import (
    DEVICE, NUM_CLASSES, DROP_RATE, USE_VIS_ONLY, USE_IR_ONLY,
    VIS_PRETRAINED_PATH, IR_PRETRAINED_PATH
)

# ================== 基础模块（与官方权重键名完全对齐） ==================
class ConvLayer(nn.Module):
    """卷积 + BN + SiLU，权重键：xxx.block.conv / block.norm / block.act"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        super().__init__()
        self.block = nn.Module()
        self.block.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                                    kernel_size//2, groups=groups, bias=False)
        self.block.norm = nn.BatchNorm2d(out_ch)
        self.block.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.block.act(self.block.norm(self.block.conv(x)))

class Conv2dOnly(nn.Module):
    """仅卷积，无 BN/激活，用于 conv_1x1 局部投影，权重键：block.conv.weight"""
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, bias=False):
        super().__init__()
        self.block = nn.Module()
        self.block.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                                    kernel_size//2, bias=bias)

    def forward(self, x):
        return self.block.conv(x)

class SE(nn.Sequential):
    """Squeeze‑and‑Excitation，直接继承 nn.Sequential，权重键：block.se.0.weight 等"""
    def __init__(self, in_ch, reduction=4):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch // reduction, in_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * super().forward(x)

class MBConvBlock(nn.Module):
    """Inverted Residual 块，权重键与官方一致"""
    def __init__(self, in_ch, out_ch, stride, expand_ratio, use_se):
        super().__init__()
        hidden_dim = int(in_ch * expand_ratio)
        self.block = nn.Module()
        if expand_ratio != 1:
            self.block.exp_1x1 = ConvLayer(in_ch, hidden_dim, 1)
        self.block.conv_3x3 = ConvLayer(hidden_dim, hidden_dim, 3, stride, groups=hidden_dim)
        if use_se:
            self.block.se = SE(hidden_dim)   # 直接使用 SE 实例，因为 SE 是 nn.Sequential，权重路径正确
        else:
            self.block.se = nn.Identity()
        self.block.red_1x1 = ConvLayer(hidden_dim, out_ch, 1)
        self.skip = (in_ch == out_ch and stride == 1)

    def forward(self, x):
        residual = x
        out = x
        if hasattr(self.block, 'exp_1x1'):
            out = self.block.exp_1x1(out)
        out = self.block.conv_3x3(out)
        out = self.block.se(out)
        out = self.block.red_1x1(out)
        if self.skip:
            out = out + residual
        return out

class MultiheadAttention(nn.Module):
    """多头注意力，权重键：in_proj_weight / in_proj_bias, out_proj.weight / out_proj.bias"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, nH, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x

class MobileViTBlockv3(nn.Module):
    """
    MobileViTv3 块，权重键完全匹配官方（conv_3x3, conv_1x1, global_rep, conv_proj, fusion）
    """
    def __init__(self, in_dim, attn_dim, out_dim, depth, mlp_dim, dropout=0.1):
        super().__init__()
        self.local_rep = nn.Module()
        self.local_rep.conv_3x3 = ConvLayer(in_dim, in_dim, 3, groups=in_dim)
        self.local_rep.conv_1x1 = Conv2dOnly(in_dim, attn_dim, 1)   # 无 BN/激活

        self.global_rep = nn.ModuleList()
        for _ in range(depth):
            block = nn.Module()
            block.pre_norm_mha = nn.Sequential(
                nn.LayerNorm(attn_dim),
                MultiheadAttention(attn_dim, 4, dropout)
            )
            block.pre_norm_ffn = nn.Sequential(
                nn.LayerNorm(attn_dim),
                nn.Linear(attn_dim, mlp_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, attn_dim),
                nn.Dropout(dropout)
            )
            self.global_rep.append(block)
        self.global_rep.append(nn.LayerNorm(attn_dim))  # 最后的 LayerNorm，键为 global_rep.{depth}.weight

        self.conv_proj = ConvLayer(attn_dim, out_dim, 1)
        self.fusion = ConvLayer(out_dim + attn_dim, out_dim, 1)   # cat(local, global_proj)

    def forward(self, x):
        # 局部特征
        local = self.local_rep.conv_3x3(x)
        local = self.local_rep.conv_1x1(local)          # (B, attn_dim, H, W)

        # 全局 Transformer：对局部特征进行建模
        B, C, H, W = local.shape
        global_feat = local.flatten(2).transpose(1, 2)  # (B, N, attn_dim)
        for i in range(len(self.global_rep) - 1):       # 前 depth 个块
            blk = self.global_rep[i]
            global_feat = global_feat + blk.pre_norm_mha[1](blk.pre_norm_mha[0](global_feat))
            global_feat = global_feat + blk.pre_norm_ffn(global_feat)
        global_feat = self.global_rep[-1](global_feat)   # 最后 LayerNorm

        # 恢复空间形状
        global_feat = global_feat.transpose(1, 2).reshape(B, C, H, W)  # (B, attn_dim, H, W)

        # 投影与融合
        global_proj = self.conv_proj(global_feat)        # (B, out_dim, H, W)
        cat = torch.cat((global_proj, local), dim=1)     # (B, out_dim+attn_dim, H, W)
        out = self.fusion(cat)                           # (B, out_dim, H, W)
        return out

class MobileViTv3_WeightCompatible(nn.Module):
    """与官方 checkpoint_ema_best.pt 严格匹配的主干网络"""
    def __init__(self, num_classes=1000, in_chans=3):
        super().__init__()
        self.conv_1 = ConvLayer(in_chans, 16, 3, 2)

        self.layer_1 = nn.Sequential(
            MBConvBlock(16, 32, 1, 4, False)
        )

        self.layer_2 = nn.Sequential(
            MBConvBlock(32, 64, 2, 4, False),
            MBConvBlock(64, 64, 1, 4, False),
            MBConvBlock(64, 64, 1, 4, False)
        )

        self.layer_3 = nn.Sequential(
            MBConvBlock(64, 128, 2, 4, False),
            MobileViTBlockv3(in_dim=128, attn_dim=144, out_dim=128, depth=2, mlp_dim=288)
        )

        self.layer_4 = nn.Sequential(
            MBConvBlock(128, 256, 2, 4, True),
            MobileViTBlockv3(in_dim=256, attn_dim=192, out_dim=256, depth=4, mlp_dim=384)
        )

        self.layer_5 = nn.Sequential(
            MBConvBlock(256, 320, 2, 4, True),
            MobileViTBlockv3(in_dim=320, attn_dim=240, out_dim=320, depth=3, mlp_dim=480)
        )

        self.conv_1x1_exp = ConvLayer(320, 960, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(960, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)


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

# ===================== 跨模态融合 =====================
class CrossModalAttention(nn.Module):
    def __init__(self, vis_feat_dim=960, ir_feat_dim=960, proj_dim=128, dropout=DROP_RATE):
        super().__init__()
        self.vis_proj = nn.Linear(vis_feat_dim, proj_dim) if vis_feat_dim != proj_dim else nn.Identity()
        self.ir_proj = nn.Linear(ir_feat_dim, proj_dim) if ir_feat_dim != proj_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.cfim = CommonFeatureInjection(proj_dim, dropout)
        self.dfim = DifferentFeatureInjection(proj_dim, dropout)

        self.self_attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=4, dropout=0.1, batch_first=True)

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


# ===================== 双主干双流模型 =====================
class DualBackboneDualStream(nn.Module):
    def __init__(self):
        super().__init__()
        # 可见光分支
        self.vis_backbone = MobileViTv3_WeightCompatible(num_classes=0, in_chans=3)
        # 红外分支（经过通道适配）
        self.ir_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1)
        self.ir_backbone = MobileViTv3_WeightCompatible(num_classes=0, in_chans=3)

        # 加载预训练权重
        self._load_weights(self.vis_backbone, VIS_PRETRAINED_PATH, "VIS")
        self._load_weights(self.ir_backbone, IR_PRETRAINED_PATH, "IR")

        # 单模态分类头
        self.vis_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(960, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(1024, NUM_CLASSES)
        )
        self.ir_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(960, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(1024, NUM_CLASSES)
        )

        # 融合模块
        self.fusion = CrossModalAttention(vis_feat_dim=960, ir_feat_dim=960, proj_dim=128, dropout=DROP_RATE)

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

        # 全微调
        self._freeze_backbones()

    def _freeze_backbones(self):
        for param in self.vis_backbone.parameters():
            param.requires_grad = True
        for param in self.ir_backbone.parameters():
            param.requires_grad = True
        print("✅ 可见光 / 红外 MobileViTv3 主干全微调")

    def _load_weights(self, model, weight_path, modal):
        if not weight_path or not os.path.exists(weight_path):
            print(f"⚠️ {modal} 预训练权重缺失：{weight_path}，随机初始化")
            return
        state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # 移除分类头权重
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith('classifier')}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️ {modal} 缺失键（将被随机初始化）: {missing}")
        if unexpected:
            print(f"⚠️ {modal} 多余键（已忽略）: {unexpected}")
        print(f"✅ {modal} 预训练权重加载成功：{weight_path}")

    def forward(self, vis_x=None, ir_x=None):
        if USE_VIS_ONLY:
            feat = self.vis_backbone(vis_x)
            out = self.vis_head(feat)
            return out, None
        elif USE_IR_ONLY:
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