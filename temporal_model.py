# # temporal_model.py
# import os
# import torch
# import torch.nn as nn
# import timm
# from config import (
#     DEVICE, NUM_CLASSES, VIS_BACKBONE_NAME, IR_BACKBONE_NAME,
#     VIS_PRETRAINED_PATH, IR_PRETRAINED_PATH, USE_VIS_ONLY, USE_IR_ONLY, DROP_RATE
# )
#
# # ---------- 门控帧内融合 ----------
# class GatedFrameFusion(nn.Module):
#     def __init__(self, vis_dim=2048, ir_dim=768, dropout=DROP_RATE):
#         super().__init__()
#         self.ir_proj = nn.Linear(ir_dim, vis_dim)   # 投影到可见光维度
#         self.gate = nn.Sequential(
#             nn.Linear(vis_dim + ir_dim, 128),
#             nn.ReLU(inplace=False),
#             nn.Dropout(dropout * 0.5),
#             nn.Linear(128, 2),
#             nn.Softmax(dim=-1)
#         )
#
#     def forward(self, vis_feat, ir_feat):
#         ir_proj = self.ir_proj(ir_feat)
#         concat = torch.cat([vis_feat, ir_feat], dim=-1)
#         weights = self.gate(concat)
#         g_vis = weights[..., 0:1]
#         g_ir  = weights[..., 1:2]
#         fused = g_vis * vis_feat + g_ir * ir_proj
#         return fused
#
# # ---------- 时间聚合模块（当前为平均池化，替代 Transformer） ----------
# class TemporalTransformer(nn.Module):
#     def __init__(self, feat_dim=2048, num_heads=8, num_layers=2, max_len=50, dropout=0.1):
#         super().__init__()
#         self.norm = nn.LayerNorm(feat_dim)
#
#     def forward(self, x, lengths):
#         # x: (B, T, D)
#         B, T, D = x.shape
#         mask = (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).float().unsqueeze(-1)
#         pooled = (x * mask).sum(dim=1) / lengths.clamp(min=1).unsqueeze(1).float()
#         return self.norm(pooled)
#
# # ---------- 时序双流模型 ----------
# class TemporalDualStream(nn.Module):
#     def __init__(self, max_seq_len=50):
#         super().__init__()
#         self.vis_backbone = timm.create_model(VIS_BACKBONE_NAME, pretrained=False, num_classes=0)
#         self.ir_backbone = timm.create_model(IR_BACKBONE_NAME, pretrained=False, num_classes=0)
#         self._load_pretrained_weights(self.vis_backbone, VIS_PRETRAINED_PATH, "VIS")
#         self._load_pretrained_weights(self.ir_backbone, IR_PRETRAINED_PATH, "IR")
#
#         self.ir_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1)
#
#         vis_dim = 2048
#         ir_dim = 768
#
#         self.frame_fusion = GatedFrameFusion(vis_dim=vis_dim, ir_dim=ir_dim, dropout=DROP_RATE)
#         self.temporal_agg = TemporalTransformer(feat_dim=vis_dim, max_len=max_seq_len, dropout=DROP_RATE*0.5)
#
#         self.classifier = nn.Sequential(
#             nn.Dropout(DROP_RATE),
#             nn.Linear(vis_dim, 1024),
#             nn.LayerNorm(1024),
#             nn.GELU(),
#             nn.Dropout(DROP_RATE * 0.8),
#             nn.Linear(1024, NUM_CLASSES)
#         )
#
#         if USE_VIS_ONLY and USE_IR_ONLY:
#             raise ValueError("不能同时开启 USE_VIS_ONLY 和 USE_IR_ONLY")
#
#         self._freeze_backbones_partially()
#
#     def _freeze_backbones_partially(self):
#         for name, param in self.vis_backbone.named_parameters():
#             if 'layer4' in name:
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False
#         for name, param in self.ir_backbone.named_parameters():
#             if 'stages.3' in name:
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False
#         print("✅ 解冻主干最后两层 (ResNet layer4, ConvNeXt stages[3])，其余冻结")
#
#     def _load_pretrained_weights(self, backbone, weight_path, modal):
#         if not weight_path or not os.path.exists(weight_path):
#             print(f"⚠️ {modal}分支权重缺失：{weight_path}，从头训练")
#             return
#         try:
#             pretrained_dict = torch.load(weight_path, map_location=DEVICE)
#             if 'model' in pretrained_dict:
#                 pretrained_dict = pretrained_dict['model']
#             if 'state_dict' in pretrained_dict:
#                 pretrained_dict = pretrained_dict['state_dict']
#             filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('head.')}
#             backbone.load_state_dict(filtered_dict, strict=False)
#             print(f"✅ {modal}分支权重加载成功：{weight_path}")
#         except Exception as e:
#             print(f"❌ {modal}分支权重加载失败：{e}，从头训练")
#
#     def forward(self, vis_seq, ir_seq, lengths):
#         B, T, C, H, W = vis_seq.shape
#         vis_flat = vis_seq.view(B * T, C, H, W)
#         ir_flat  = ir_seq.view(B * T, 1, H, W)
#
#         vis_feat = self.vis_backbone(vis_flat)
#         ir_adapted = self.ir_adapter(ir_flat)
#         ir_feat = self.ir_backbone(ir_adapted)
#
#         vis_feat = vis_feat.view(B, T, -1)
#         ir_feat  = ir_feat.view(B, T, -1)
#
#         fused = self.frame_fusion(vis_feat, ir_feat)
#         global_feat = self.temporal_agg(fused, lengths)
#
#         logits = self.classifier(global_feat)
#         return logits, global_feat



# import os
# import torch
# import torch.nn as nn
# import timm
# from config import (
#     DEVICE, NUM_CLASSES, VIS_BACKBONE_NAME, IR_BACKBONE_NAME,
#     VIS_PRETRAINED_PATH, IR_PRETRAINED_PATH, USE_VIS_ONLY, USE_IR_ONLY, DROP_RATE
# )
#
# # -------------------- 门控帧内融合 --------------------
# class GatedFrameFusion(nn.Module):
#     def __init__(self, vis_dim=2048, ir_dim=768, dropout=DROP_RATE):
#         super().__init__()
#         self.ir_proj = nn.Linear(ir_dim, vis_dim)
#         self.gate = nn.Sequential(
#             nn.Linear(vis_dim + ir_dim, 128),
#             nn.ReLU(inplace=False),
#             nn.Dropout(dropout * 0.5),
#             nn.Linear(128, 2),
#             nn.Softmax(dim=-1)
#         )
#
#     def forward(self, vis_feat, ir_feat):
#         ir_proj = self.ir_proj(ir_feat)
#         concat = torch.cat([vis_feat, ir_feat], dim=-1)
#         weights = self.gate(concat)
#         g_vis = weights[..., 0:1]
#         g_ir  = weights[..., 1:2]
#         fused = g_vis * vis_feat + g_ir * ir_proj
#         return fused
#
# # -------------------- 稳定版双向 LSTM（无打包，直接处理） --------------------
# class TemporalLSTM(nn.Module):
#     def __init__(self, feat_dim=2048, hidden_dim=512, num_layers=1, dropout=0.1):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_size=feat_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             bidirectional=True,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )
#         self.fc = nn.Linear(hidden_dim * 2, feat_dim)
#         self.norm = nn.LayerNorm(feat_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, lengths):
#         B, T, D = x.shape
#         # 直接喂入 LSTM，不打包
#         out, _ = self.lstm(x)                      # (B, T, hidden_dim*2)
#
#         # 每个样本取最后一个有效帧
#         mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)   # (B, T)
#         # 找到每个样本最后一个 True 的索引
#         mask_flip = mask.flip(1)
#         last_idx = T - 1 - mask_flip.float().argmax(dim=1)   # (B,)
#         last = out[torch.arange(B), last_idx]                # (B, hidden_dim*2)
#
#         last = self.fc(last)
#         last = self.dropout(last)
#         return self.norm(last)
#
# # -------------------- 时序双流模型 --------------------
# class TemporalDualStream(nn.Module):
#     def __init__(self, max_seq_len=50):
#         super().__init__()
#         self.vis_backbone = timm.create_model(VIS_BACKBONE_NAME, pretrained=False, num_classes=0)
#         self.ir_backbone = timm.create_model(IR_BACKBONE_NAME, pretrained=False, num_classes=0)
#         self._load_pretrained_weights(self.vis_backbone, VIS_PRETRAINED_PATH, "VIS")
#         self._load_pretrained_weights(self.ir_backbone, IR_PRETRAINED_PATH, "IR")
#
#         self.ir_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1)
#         vis_dim, ir_dim = 2048, 768
#
#         self.frame_fusion = GatedFrameFusion(vis_dim=vis_dim, ir_dim=ir_dim)
#         self.temporal_agg = TemporalLSTM(feat_dim=vis_dim, hidden_dim=512, dropout=DROP_RATE*0.5)
#         self.classifier = nn.Sequential(
#             nn.Dropout(DROP_RATE),
#             nn.Linear(vis_dim, 1024),
#             nn.LayerNorm(1024),
#             nn.GELU(),
#             nn.Dropout(DROP_RATE * 0.8),
#             nn.Linear(1024, NUM_CLASSES)
#         )
#         if USE_VIS_ONLY and USE_IR_ONLY:
#             raise ValueError("不能同时开启 USE_VIS_ONLY 和 USE_IR_ONLY")
#         self._freeze_backbones_partially()
#
#     def _freeze_backbones_partially(self):
#         for n, p in self.vis_backbone.named_parameters():
#             p.requires_grad = 'layer4' in n
#         for n, p in self.ir_backbone.named_parameters():
#             p.requires_grad = 'stages.3' in n
#         print("✅ 解冻主干最后两层")
#
#     def _load_pretrained_weights(self, backbone, path, modal):
#         if not path or not os.path.exists(path):
#             print(f"⚠️ {modal}权重缺失")
#             return
#         try:
#             ckpt = torch.load(path, map_location=DEVICE)
#             ckpt = ckpt.get('model', ckpt).get('state_dict', ckpt)
#             filtered = {k: v for k, v in ckpt.items() if 'head.' not in k}
#             backbone.load_state_dict(filtered, strict=False)
#             print(f"✅ {modal}权重加载成功")
#         except Exception as e:
#             print(f"❌ {modal}权重加载失败: {e}")
#
#     def forward(self, vis_seq, ir_seq, lengths):
#         B, T, C, H, W = vis_seq.shape
#         vis_flat = vis_seq.view(B*T, C, H, W)
#         ir_flat  = ir_seq.view(B*T, 1, H, W)
#
#         vis_feat = self.vis_backbone(vis_flat)
#         ir_adapted = self.ir_adapter(ir_flat)
#         ir_feat   = self.ir_backbone(ir_adapted)
#
#         vis_feat = vis_feat.view(B, T, -1)
#         ir_feat  = ir_feat.view(B, T, -1)
#
#         fused = self.frame_fusion(vis_feat, ir_feat)
#         global_feat = self.temporal_agg(fused, lengths)
#         logits = self.classifier(global_feat)
#         return logits, global_feat

# temporal_model.py
import os
import torch
import torch.nn as nn
import timm
from config import (
    DEVICE, NUM_CLASSES, VIS_BACKBONE_NAME, IR_BACKBONE_NAME,
    VIS_PRETRAINED_PATH, IR_PRETRAINED_PATH, USE_VIS_ONLY, USE_IR_ONLY, DROP_RATE
)

# -------------------- 门控帧内融合 (不变) --------------------
class GatedFrameFusion(nn.Module):
    def __init__(self, vis_dim=2048, ir_dim=768, dropout=DROP_RATE):
        super().__init__()
        self.ir_proj = nn.Linear(ir_dim, vis_dim)
        self.gate = nn.Sequential(
            nn.Linear(vis_dim + ir_dim, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, vis_feat, ir_feat):
        ir_proj = self.ir_proj(ir_feat)
        concat = torch.cat([vis_feat, ir_feat], dim=-1)
        weights = self.gate(concat)
        g_vis = weights[..., 0:1]
        g_ir  = weights[..., 1:2]
        fused = g_vis * vis_feat + g_ir * ir_proj
        return fused

# -------------------- 超轻量时序 LSTM --------------------
class TemporalLSTM(nn.Module):
    def __init__(self, input_dim=2048, bottleneck_dim=128, hidden_dim=128, num_layers=1, dropout=0.5):
        super().__init__()
        # 降维层
        self.fc_down = nn.Linear(input_dim, bottleneck_dim)
        self.ln_down = nn.LayerNorm(bottleneck_dim)
        self.dropout_down = nn.Dropout(dropout)
        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=bottleneck_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # 输出投影
        self.fc_out = nn.Linear(hidden_dim * 2, bottleneck_dim)
        self.norm = nn.LayerNorm(bottleneck_dim)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, x, lengths):
        B, T, D = x.shape
        # 降维
        x = self.dropout_down(self.ln_down(self.fc_down(x)))   # (B, T, bottleneck_dim)
        # LSTM 前向
        out, _ = self.lstm(x)                                   # (B, T, hidden_dim*2)
        # 取每个样本的最后一个有效帧
        last_idx = (lengths - 1).clamp(min=0).long()            # (B,)
        last = out[torch.arange(B), last_idx]                   # (B, hidden_dim*2)
        # 投影 + 归一化
        last = self.dropout_out(self.fc_out(last))
        return self.norm(last)

# -------------------- 时序双流模型 --------------------
class TemporalDualStream(nn.Module):
    def __init__(self, max_seq_len=50):
        super().__init__()
        # 主干网络
        self.vis_backbone = timm.create_model(VIS_BACKBONE_NAME, pretrained=False, num_classes=0)
        self.ir_backbone = timm.create_model(IR_BACKBONE_NAME, pretrained=False, num_classes=0)
        self._load_pretrained_weights(self.vis_backbone, VIS_PRETRAINED_PATH, "VIS")
        self._load_pretrained_weights(self.ir_backbone, IR_PRETRAINED_PATH, "IR")

        # 红外通道适配
        self.ir_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1)

        vis_dim, ir_dim = 2048, 768
        # 帧内融合
        self.frame_fusion = GatedFrameFusion(vis_dim=vis_dim, ir_dim=ir_dim, dropout=DROP_RATE)
        # 时序聚合：轻量 LSTM
        self.temporal_agg = TemporalLSTM(
            input_dim=vis_dim,
            bottleneck_dim=128,
            hidden_dim=128,
            num_layers=1,
            dropout=0.5          # 用较高的 dropout 防止过拟合
        )

        # 分类头（输入维度为 bottleneck_dim = 128）
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, NUM_CLASSES)
        )

        if USE_VIS_ONLY and USE_IR_ONLY:
            raise ValueError("不能同时开启 USE_VIS_ONLY 和 USE_IR_ONLY")

        # 部分冻结主干
        self._freeze_backbones_partially()

    def _freeze_backbones_partially(self):
        for n, p in self.vis_backbone.named_parameters():
            p.requires_grad = 'layer4' in n
        for n, p in self.ir_backbone.named_parameters():
            p.requires_grad = 'stages.3' in n
        print("✅ 解冻主干最后两层 (ResNet layer4, ConvNeXt stages[3])，其余冻结")

    def _load_pretrained_weights(self, backbone, weight_path, modal):
        if not weight_path or not os.path.exists(weight_path):
            print(f"⚠️ {modal}分支权重缺失：{weight_path}，从头训练")
            return
        try:
            pretrained_dict = torch.load(weight_path, map_location=DEVICE)
            # 处理可能被包装的状态字典
            if 'model' in pretrained_dict:
                pretrained_dict = pretrained_dict['model']
            if 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']
            # 去除分类头对应的权重
            filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('head.')}
            backbone.load_state_dict(filtered_dict, strict=False)
            print(f"✅ {modal}分支权重加载成功：{weight_path}")
        except Exception as e:
            print(f"❌ {modal}分支权重加载失败：{e}，从头训练")

    def forward(self, vis_seq, ir_seq, lengths):
        """
        vis_seq: (B, T, 3, H, W)
        ir_seq:  (B, T, 1, H, W)
        lengths: (B,) 每个样本的实际帧数
        """
        B, T, C, H, W = vis_seq.shape

        # 展平时间与批次维度，以便并行通过主干网络
        vis_flat = vis_seq.view(B * T, C, H, W)
        ir_flat  = ir_seq.view(B * T, 1, H, W)

        # 提取每帧特征
        vis_feat = self.vis_backbone(vis_flat)          # (B*T, 2048)
        ir_adapted = self.ir_adapter(ir_flat)           # (B*T, 3, H, W)
        ir_feat = self.ir_backbone(ir_adapted)          # (B*T, 768)

        # 恢复时间维度
        vis_feat = vis_feat.view(B, T, -1)              # (B, T, 2048)
        ir_feat  = ir_feat.view(B, T, -1)               # (B, T, 768)

        # 帧内门控融合
        fused = self.frame_fusion(vis_feat, ir_feat)    # (B, T, 2048)

        # 时间聚合（使用 LSTM）
        global_feat = self.temporal_agg(fused, lengths) # (B, 128)

        # 分类
        logits = self.classifier(global_feat)           # (B, NUM_CLASSES)

        return logits, global_feat