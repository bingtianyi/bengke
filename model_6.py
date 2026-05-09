import torch
import os
import torch.nn as nn
import timm
from config import (
    DEVICE, NUM_CLASSES, VIS_BACKBONE_NAME, IR_BACKBONE_NAME,
    VIS_PRETRAINED_PATH, IR_PRETRAINED_PATH, USE_VIS_ONLY, USE_IR_ONLY, DROP_RATE
)

# ===================== 基础组件定义（MobileViTv3-S 所需） =====================
class ConvBlock(nn.Module):
    """ 对应官方 ConvLayer，内部 block 包含 conv, norm, act """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        super().__init__()
        self.block = nn.Module()
        self.block.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, kernel_size//2, groups=groups, bias=False)
        self.block.norm = nn.BatchNorm2d(out_ch)
        self.block.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.block.act(self.block.norm(self.block.conv(x)))

class MBConvBlock(nn.Module):
    """ 对应官方 MBConv，内部子模块命名与权重一致 """
    def __init__(self, in_ch, out_ch, stride, expand_ratio, use_se):
        super().__init__()
        hidden_dim = int(in_ch * expand_ratio)
        self.block = nn.Module()
        if expand_ratio != 1:
            self.block.exp_1x1 = ConvBlock(in_ch, hidden_dim, 1)
        self.block.conv_3x3 = ConvBlock(hidden_dim, hidden_dim, 3, stride, groups=hidden_dim)
        if use_se:
            self.block.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, hidden_dim, 1),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.block.se = nn.Identity()
        self.block.red_1x1 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.skip = (in_ch == out_ch and stride == 1)

    def forward(self, x):
        residual = x
        out = x
        if hasattr(self.block, 'exp_1x1'):
            out = self.block.exp_1x1(out)
        out = self.block.conv_3x3(out)
        out = out * self.block.se(out)
        out = self.block.red_1x1(out)
        if self.skip:
            out = out + residual
        return out

class MobileViTv3Block(nn.Module):
    def __init__(self, dim, depth, kernel_size, patch_size, mlp_dim):
        super().__init__()
        self.local_rep = nn.Module()
        self.local_rep.conv_3x3 = ConvBlock(dim, dim, 3, groups=dim)
        self.local_rep.conv_1x1 = ConvBlock(dim, dim, 1)

        self.global_rep = nn.ModuleList()
        for _ in range(depth):
            layer = nn.Module()
            layer.pre_norm_mha = nn.Sequential(
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, 4, batch_first=True)
            )
            layer.pre_norm_ffn = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(mlp_dim, dim),
                nn.Dropout(0.1)
            )
            self.global_rep.append(layer)

        self.conv_proj = ConvBlock(dim, dim, 1)
        self.fusion = ConvBlock(dim, dim, 3)

    def forward(self, x):
        # 局部表示
        local = self.local_rep.conv_3x3(x)
        local = self.local_rep.conv_1x1(local)

        # 全局表示 (Transformer)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)          # (B, N, C)
        for blk in self.global_rep:
            norm_x = blk.pre_norm_mha[0](x)       # LayerNorm
            attn_out, _ = blk.pre_norm_mha[1](norm_x, norm_x, norm_x)  # 自注意力
            x = x + attn_out                       # 残差
            x = x + blk.pre_norm_ffn(x)            # FFN + 残差
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # 融合
        x = self.conv_proj(x)
        x = self.fusion(x + local)
        return x

class MobileViTv3_S_Official(nn.Module):
    def __init__(self, num_classes=0):
        super().__init__()
        self.conv_1 = ConvBlock(3, 16, 3, 2)

        self.layer_1 = nn.Sequential(
            MBConvBlock(16, 16, 1, 1, False)
        )

        self.layer_2 = nn.Sequential(
            MBConvBlock(16, 32, 2, 4, False),
            MBConvBlock(32, 32, 1, 4, False),
            MBConvBlock(32, 32, 1, 4, False)
        )

        self.layer_3 = nn.Sequential(
            MBConvBlock(32, 64, 2, 4, False),
            MobileViTv3Block(64, 2, 3, 2, 128)
        )

        self.layer_4 = nn.Sequential(
            MBConvBlock(64, 128, 2, 4, True),
            MobileViTv3Block(128, 4, 3, 2, 256)
        )

        self.layer_5 = nn.Sequential(
            MBConvBlock(128, 256, 2, 4, True),
            MobileViTv3Block(256, 3, 3, 2, 512)
        )

        self.conv_1x1_exp = ConvBlock(256, 640, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(640, num_classes) if num_classes > 0 else nn.Identity()

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


# ===================== 跨模态融合（带自注意力增强） =====================
class CrossModalAttention(nn.Module):
    def __init__(self, vis_feat_dim=640, ir_feat_dim=768, proj_dim=128, dropout=DROP_RATE):
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

        # ---- 可见光分支：MobileViTv3-S，随机初始化，全微调 ----
        self.vis_backbone = MobileViTv3_S_Official(num_classes=0)   # 输出 640 维特征
        # 不再加载预训练权重

        # ---- 红外分支：保持原样，部分冻结 ----
        self.ir_backbone = timm.create_model(IR_BACKBONE_NAME, pretrained=False, num_classes=0)
        self._load_pretrained_weights(self.ir_backbone, IR_PRETRAINED_PATH, "IR")
        self.ir_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1)

        # 单模态分类头（备用）
        self.vis_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(640, 1024),          # 输入维度改为 640
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(1024, NUM_CLASSES)
        )
        self.ir_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(512, NUM_CLASSES)
        )

        # 融合模块：注意 vis_feat_dim=640
        self.fusion = CrossModalAttention(vis_feat_dim=640, ir_feat_dim=768, proj_dim=128, dropout=DROP_RATE)

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

        # 部分冻结主干：可见光全微调，红外只解冻最后两层
        self._freeze_backbones_partially()

    def _freeze_backbones_partially(self):
        """ 可见光 MobileViTv3-S 全微调；红外 ConvNeXtV2 仅解冻 stages[3] """
        # 可见光分支：所有参数可训练
        for param in self.vis_backbone.parameters():
            param.requires_grad = True

        # 红外分支：冻结前部，只解冻 stages[3]
        for name, param in self.ir_backbone.named_parameters():
            if 'stages.3' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("✅ 可见光 MobileViTv3-S 全微调，红外仅解冻 stages[3]")

    def _load_pretrained_weights(self, backbone, weight_path, modal):
        if not weight_path or not os.path.exists(weight_path):
            print(f"⚠️ {modal}分支权重缺失：{weight_path}，从头训练")
            return
        try:
            pretrained_dict = torch.load(weight_path, map_location=DEVICE, weights_only=False)
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
            return out, None
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