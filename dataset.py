#1
import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from config import (
    DEVICE, NUM_CLASSES, USE_VIS_ONLY, USE_IR_ONLY, DROP_RATE,
    TARGET_DATA_ROOT, IMAGE_SIZE, USE_LIGHT_FILTER, LIGHT_SCENARIO,
    EMOTION_MAP
)

# ------------------ 大小写不敏感路径查找 ------------------
def find_path_insensitive(path):
    """
    逐级查找大小写不敏感的完整路径。
    返回真实存在的路径字符串，若找不到则返回 None。
    """
    path = os.path.normpath(path)
    parts = path.split(os.sep)
    if os.path.isabs(path):
        current = parts[0] + os.sep
        parts = parts[1:]
    else:
        current = ''
    for part in parts:
        if not os.path.exists(current):
            return None
        try:
            items = os.listdir(current)
        except (PermissionError, FileNotFoundError):
            return None
        found = None
        for item in items:
            if item.lower() == part.lower():
                found = item
                break
        if found is None:
            return None
        current = os.path.join(current, found)
    return current

def safe_open_image(path, mode='RGB'):
    """鲁棒打开图像，支持整个路径的大小写不敏感和扩展名自动尝试"""
    # 1. 直接尝试原路径
    if os.path.exists(path):
        return Image.open(path).convert(mode)
    # 2. 大小写不敏感路径查找
    real_path = find_path_insensitive(path)
    if real_path:
        return Image.open(real_path).convert(mode)
    # 3. 尝试替换扩展名 + 大小写不敏感
    base, ext = os.path.splitext(path)
    for alt_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        alt_path = base + alt_ext
        real_path = find_path_insensitive(alt_path)
        if real_path:
            return Image.open(real_path).convert(mode)
    raise FileNotFoundError(f"Image not found: {path}")

#
class SyncTransform:
    def __init__(self, rgb_color_jitter=None, rotation_deg=15, hflip_prob=0.5):
        self.rgb_color_jitter = rgb_color_jitter
        self.rotation_deg = rotation_deg
        self.hflip_prob = hflip_prob
        self.crop_scale = (0.90, 1.0)      # 85%~100% 区域

    def __call__(self, rgb_img, ir_img):
        seed = random.randint(0, 2**32)
        random.seed(seed)

        # 1. 同步随机裁剪并 resize
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            rgb_img, scale=self.crop_scale, ratio=(0.9, 1.1)
        )
        rgb_img = F.resized_crop(rgb_img, i, j, h, w, (IMAGE_SIZE, IMAGE_SIZE))
        ir_img = F.resized_crop(ir_img, i, j, h, w, (IMAGE_SIZE, IMAGE_SIZE))

        # 2. 随机水平翻转
        if random.random() < self.hflip_prob:
            rgb_img = F.hflip(rgb_img)
            ir_img = F.hflip(ir_img)

        # 3. 随机旋转（角度范围扩大）
        if self.rotation_deg > 0:
            angle = random.uniform(-self.rotation_deg, self.rotation_deg)
            rgb_img = F.rotate(rgb_img, angle)
            ir_img = F.rotate(ir_img, angle)

        # 4. 颜色抖动（仅 VIS）
        if self.rgb_color_jitter is not None:
            rgb_img = self.rgb_color_jitter(rgb_img)

        return rgb_img, ir_img

# ------------------ 数据集类 ------------------
class OuluCASIADataset(Dataset):
    def __init__(self, split='train', light_scenario=None):
        self.root = TARGET_DATA_ROOT
        self.split = split
        self.light_scenario = light_scenario if light_scenario is not None else LIGHT_SCENARIO
        self.label_to_idx = EMOTION_MAP

        csv_filename = f'{split}.csv'
        self.annotation_path = os.path.join(self.root, csv_filename)
        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(f"❌ 标注文件缺失：{self.annotation_path}")

        self.annotation = pd.read_csv(self.annotation_path)

        if USE_LIGHT_FILTER and self.light_scenario is not None:
            self.annotation = self.annotation[
                self.annotation['vis_path'].str.contains(self.light_scenario, case=False, na=False)
            ]

        print(f"📊 {split} 集 {self.light_scenario if self.light_scenario else '全部'} 样本数：{len(self.annotation)}")
        self.transform = self._get_transform()

    def _get_transform(self):
        if self.split == 'train':
            rgb_color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            sync_geo = SyncTransform(
                rgb_color_jitter=rgb_color_jitter,
                rotation_deg=10,
                hflip_prob=0.5
            )
            rgb_post = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            ir_post = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            return (sync_geo, rgb_post, ir_post)
        else:
            resize_size = 256 if IMAGE_SIZE <= 256 else int(IMAGE_SIZE * 1.1)
            rgb_post = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            ir_post = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            return (None, rgb_post, ir_post)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        row = self.annotation.iloc[idx]

        vis_path = os.path.join(self.root, row['vis_path'])
        ir_path = os.path.join(self.root, row['ir_path'])
        label_str = row['label']

        # 标签转换
        if isinstance(label_str, str):
            if label_str == "Happy":
                label_str = "Happiness"
            elif label_str == "Sad":
                label_str = "Sadness"
            label = self.label_to_idx.get(label_str, -1)
            if label == -1:
                raise ValueError(f"未知标签 '{label_str}'，请检查 EMOTION_MAP")
        else:
            label = int(label_str)

        # 鲁棒打开图像（支持全路径大小写不敏感）
        vis_img = safe_open_image(vis_path, 'RGB')
        ir_img = safe_open_image(ir_path, 'L')

        sync_geo, rgb_post, ir_post = self.transform
        if sync_geo is not None:
            vis_img, ir_img = sync_geo(vis_img, ir_img)

        vis_img = rgb_post(vis_img)
        ir_img = ir_post(ir_img)

        return vis_img, ir_img, label

# ------------------ 测试 ------------------
if __name__ == '__main__':
    print(f"📌 数据集根目录：{TARGET_DATA_ROOT}")
    print(f"📌 USE_LIGHT_FILTER = {USE_LIGHT_FILTER}, LIGHT_SCENARIO = {LIGHT_SCENARIO}")

    try:
        train_all = OuluCASIADataset('train')
        if len(train_all) > 0:
            vis, ir, label = train_all[0]
            print(f"✅ 训练集全部加载成功 | {vis.shape} | {ir.shape} | label={label}")
    except Exception as e:
        print(f"❌ 训练集全部失败：{e}")

    if USE_LIGHT_FILTER:
        try:
            train_weak = OuluCASIADataset('train', 'weak')
            if len(train_weak) > 0:
                print(f"✅ 训练集 weak 场景加载成功")
        except Exception as e:
            print(f"❌ 训练集 weak 失败：{e}")

        try:
            val_dark = OuluCASIADataset('val', 'dark')
            if len(val_dark) > 0:
                print(f"✅ 验证集 dark 场景加载成功")
        except Exception as e:
            print(f"❌ 验证集 dark 失败：{e}")