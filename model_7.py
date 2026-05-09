import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import (
    DEVICE, NUM_CLASSES, VIS_BACKBONE_NAME, IR_BACKBONE_NAME,
    VIS_PRETRAINED_PATH, IR_PRETRAINED_PATH, USE_VIS_ONLY, USE_IR_ONLY, DROP_RATE
)

# ===================== FER-VMamba 自定义模块（全局分支） =====================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(nn.Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, split_out_channels, stride):
        super(MDConv, self).__init__()
        self.num_groups = len(kernel_size)
        self.split_channels = split_out_channels
        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv2d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i] // 2,
                groups=self.split_channels[i],
                bias=False
            ))
        self.bn = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)
        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)
        return x

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // groups)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        y = identity * x_w * x_h
        return y

class Mix_Depth_Wise(nn.Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1),
                 groups=1, kernel_size=[3, 5, 7], split_out_channels=[64, 32, 32]):
        super(Mix_Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = MDConv(channels=groups, kernel_size=kernel_size,
                              split_out_channels=split_out_channels, stride=stride)
        self.CA = CoordAtt(groups, groups)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.CA(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Mix_Residual(nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1),
                 kernel_size=[3, 5], split_out_channels=[64, 64]):
        super(Mix_Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Mix_Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride,
                               groups=groups, kernel_size=kernel_size, split_out_channels=split_out_channels)
            )
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

class ChannelSpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelSpatialAttentionBlock, self).__init__()
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.Conv2d(2, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se_attention = self.se_block(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_feature = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.sigmoid(self.conv1x1(spatial_feature))
        output = se_attention * x * spatial_attention
        return output

# ===================== FER-VMamba 可见光主干（支持任意输入尺寸） =====================

class VisBackbone_FERVMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Mix_Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1),
                                      groups=128, kernel_size=[3, 5, 7], split_out_channels=[64, 32, 32])
        self.conv_3 = Mix_Residual(64, num_block=9, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1),
                                   kernel_size=[3, 5], split_out_channels=[96, 32])
        self.conv_34 = Mix_Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1),
                                      groups=256, kernel_size=[3, 5, 7], split_out_channels=[128, 64, 64])
        self.conv_4 = Mix_Residual(128, num_block=16, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1),
                                   kernel_size=[3, 5], split_out_channels=[192, 64])
        self.conv_45 = Mix_Depth_Wise(128, 256, kernel=(3, 3), stride=(2, 2), padding=(1, 1),
                                      groups=512 * 2, kernel_size=[3, 5, 7, 9],
                                      split_out_channels=[128 * 2, 128 * 2, 128 * 2, 128 * 2])
        self.conv_5 = Mix_Residual(256, num_block=6, groups=512, kernel=(3, 3), stride=(1, 1), padding=(1, 1),
                                   kernel_size=[3, 5, 7], split_out_channels=[86 * 2, 85 * 2, 85 * 2])
        self.attention = ChannelSpatialAttentionBlock(256)
        self.conv_6_sep = Conv_block(256, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.attention(out)
        out = self.conv_6_sep(out)
        out = self.pool(out)
        out = self.flatten(out)
        return out   # (B, 512)


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
    def __init__(self, vis_feat_dim=512, ir_feat_dim=768, proj_dim=128, dropout=DROP_RATE):
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
        fused_feat = self.fusion_mlp(concat_feat)          # (B, proj_dim)

        fused_feat = fused_feat.unsqueeze(1)               # (B, 1, proj_dim)
        attn_out, _ = self.self_attn(fused_feat, fused_feat, fused_feat)
        fused_feat = attn_out.squeeze(1)                   # (B, proj_dim)

        return fused_feat

# ===================== 双主干双流模型 =====================
class DualBackboneDualStream(nn.Module):
    def __init__(self):
        super().__init__()
        # 可见光分支：FER-VMamba 全局特征提取（支持任意输入尺寸，随机初始化）
        self.vis_backbone = VisBackbone_FERVMamba()

        # 红外分支：保持原样，部分冻结
        self.ir_backbone = timm.create_model(IR_BACKBONE_NAME, pretrained=False, num_classes=0)
        self._load_pretrained_weights(self.ir_backbone, IR_PRETRAINED_PATH, "IR")
        self.ir_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1)

        # 单模态分类头（备用）
        self.vis_head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(512, 1024),          # 输入 512（FER-VMamba 特征维度）
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

        # 融合模块：vis 特征维度 512
        self.fusion = CrossModalAttention(vis_feat_dim=512, ir_feat_dim=768, proj_dim=128, dropout=DROP_RATE)

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

        # 部分冻结主干：可见光全微调，红外仅解冻最后两层
        self._freeze_backbones_partially()

    def _freeze_backbones_partially(self):
        """ 可见光 FER-VMamba 全微调；红外 ConvNeXtV2 仅解冻 stages[3] """
        for param in self.vis_backbone.parameters():
            param.requires_grad = True

        for name, param in self.ir_backbone.named_parameters():
            if 'stages.3' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("✅ 可见光 FER-VMamba 全微调，红外仅解冻 stages[3]")

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