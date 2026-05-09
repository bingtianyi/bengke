import argparse
import torch
from cvnets.models.classification.mobilevit_v3 import MobileViTv3

# 1. 构建参数解析器并配置MobileViTv3-S参数
parser = argparse.ArgumentParser()
parser = MobileViTv3.add_arguments(parser)  # 加载模型专属参数

# 设置核心参数（MobileViTv3-S）
args = parser.parse_args([
    "--model.classification.mitv3.width-multiplier", "1.0",
    "--model.classification.mitv3.attn-dropout", "0.0",
    "--model.classification.mitv3.ffn-dropout", "0.0",
    "--model.classification.mitv3.dropout", "0.0",
    "--model.classification.n_classes", "1000",  # 预训练权重通常对应ImageNet-1k
    "--model.layer.global_pool", "mean"
])

# 2. 实例化MobileViTv3-S模型
model = MobileViTv3(opts=args)
model.eval()  # 推理时启用eval模式（关闭dropout/bn训练行为）
print("MobileViTv3-S模型实例化完成，模型结构：", model)