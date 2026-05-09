#
import os
import torch

# ===================== 路径配置 =====================
RAW_DATA_ROOT = r"E:\pycharm_lianxi\renlianqingxushibie\data\Oulu_CASIA_NIR_VIS"
#TARGET_DATA_ROOT = r"/root/autodl-tmp/renlianqingxushibie/data/oulu"
#TARGET_DATA_ROOT =r"data/oulu"

ORIGINAL_DATA_ROOT = r"data/oulu_aug_test4"
#ORIGINAL_DATA_ROOT = r"/root/autodl-tmp/renlianqingxushibie/data/oulu_aug_test4"
# 增强合并后的数据集根目录（包含 train.csv / val.csv，且目录结构为 split/modal/emotion/）
#AUG_MERGED_DATA_ROOT = r"/root/autodl-tmp/renlianqingxushibie/data/oulu_aug_test4"
AUG_MERGED_DATA_ROOT = r"data/oulu_aug_test4"

# 光照场景配置（可选：dark/strong/weak）
#LIGHT_SCENARIO = "weak"  # 可切换为 dark/strong/weak
LIGHT_SCENARIO = "weak"  # 表示不再按光照筛选\


# 根据 LIGHT_SCENARIO 动态选择数据集根目录和模式
if LIGHT_SCENARIO is None:
    TARGET_DATA_ROOT = AUG_MERGED_DATA_ROOT
    USE_LIGHT_FILTER = False   # 不使用光照筛选（直接读取 CSV）
else:
    TARGET_DATA_ROOT = ORIGINAL_DATA_ROOT
    USE_LIGHT_FILTER = True    # 使用光照筛选（按目录结构加载）

OUTPUT_DIR = './output'           # 输出配置
LOG_DIR = "./logs"                # 训练日志路径
CSV_TRAIN_PATH = os.path.join(TARGET_DATA_ROOT, "train.csv")
CSV_VAL_PATH = os.path.join(TARGET_DATA_ROOT, "val.csv")

# ===================== 数据配置 =====================
EMOTION_MAP = {"Anger":0, "Disgust":1, "Fear":2, "Happiness":3, "Sadness":4, "Surprise":5}
NUM_CLASSES = 6  # 6种情绪
IMAGE_SIZE = 224  # 统一变量名（和代码中IMAGE_SIZE对齐）
BATCH_SIZE = 128
NUM_WORKERS = 0  # 建议设0，避免跨平台/多线程问题
FRAME_NUM_START = 0
REQUIRED_FRAME_NUM = 3


# ===================== 训练配置 =====================
EPOCHS = 50
LR = 5e-4  # 统一变量名LEARNING_RATE
WEIGHT_DECAY = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_VAL_SPLIT = 0.8
EARLY_STOPPING_PATIENCE = 20
DROP_RATE = 0.5
IR_BACKBONE_LR_RATIO = 0.5   # 新增：红外主干学习率倍数（可设为 0.5~1.0）

# ===================== 模型配置 =====================
# 双主干配置：分别指定VIS/IR的主干
VIS_BACKBONE_NAME = 'resnet50'       # 可见光分支：ResNet50
IR_BACKBONE_NAME = 'convnextv2_tiny' # 红外分支：ConvNeXtV2-Tiny
# VIS_PRETRAINED_PATH = 'pretrained/mobilevitv3_s.pth'   # 或 'pretrained/mobilevitv3_s.pth'
# IR_PRETRAINED_PATH = 'pretrained/mobilevitv3_s.pth'    # 红外可复用相同权重（或留空）
# # 双主干预训练权重路径
VIS_PRETRAINED_PATH = './pretrained/resnet50.pth'
IR_PRETRAINED_PATH = './pretrained/convnextv2_tiny.pth'
#IR_PRETRAINED_PATH = './pretrained/resnet50.pth'

# 单模态开关（二选一，不能同时为True）
USE_VIS_ONLY = False # 仅可见光：设True，IR设False
USE_IR_ONLY = False  # 仅红外：设True，VIS设False






