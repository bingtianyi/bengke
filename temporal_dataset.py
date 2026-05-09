import os
import random
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.transforms import functional as F
from config import IMAGE_SIZE, EMOTION_MAP


class TemporalOuluCASIADataset(Dataset):
    """
    时序红外‑可见光人脸表情数据集 (适配变长序列)

    读取预处理生成的序列 JSON (直接是列表)，每次返回一个完整的帧序列
    (或按需截取/补齐)，并对序列内所有帧应用同步几何增强。

    Args:
        json_path: JSON 索引文件路径 (如 'temporal_data/train_sequences.json')
        data_root: 图像文件所在的根目录，默认与 JSON 所在目录相同
        seq_len: 序列长度，设为 None 则使用序列的完整长度，否则按指定长度截取或补齐
        rgb_color_jitter: 颜色抖动变换 (仅用于可见光)
        rotation_deg: 随机旋转的最大角度
        hflip_prob: 随机水平翻转概率
    """
    def __init__(self, json_path, data_root=None, seq_len=None,
                 rgb_color_jitter=None, rotation_deg=10, hflip_prob=0.5):
        with open(json_path, 'r') as f:
            self.sequences = json.load(f)          # 直接是列表
        self.seq_len = seq_len
        self.rgb_color_jitter = rgb_color_jitter
        self.rotation_deg = rotation_deg
        self.hflip_prob = hflip_prob

        # 数据根目录：默认与 JSON 文件所在目录相同
        if data_root is None:
            self.data_root = os.path.dirname(os.path.abspath(json_path))
        else:
            self.data_root = data_root

        # 标签映射
        self.label_to_idx = EMOTION_MAP

        # 基础后处理：ToTensor + 归一化
        self.vis_post = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.ir_post = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.sequences)

    def _sync_geometry(self, vis_img, ir_img, seed):
        """对一对可见光‑红外图像施加同步的几何变换（翻转+旋转）"""
        random.seed(seed)
        # 缩放到固定尺寸（如果已经是 224x224，这里只做安全转换）
        vis_img = F.resize(vis_img, (IMAGE_SIZE, IMAGE_SIZE))
        ir_img = F.resize(ir_img, (IMAGE_SIZE, IMAGE_SIZE))

        if random.random() < self.hflip_prob:
            vis_img = F.hflip(vis_img)
            ir_img = F.hflip(ir_img)

        if self.rotation_deg > 0:
            angle = random.uniform(-self.rotation_deg, self.rotation_deg)
            vis_img = F.rotate(vis_img, angle)
            ir_img = F.rotate(ir_img, angle)

        return vis_img, ir_img

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        vis_frames = seq['vis_frames']      # 文件相对路径列表（已按帧号排序）
        ir_frames  = seq['ir_frames']
        emotion_str = seq['emotion']

        # 标签转换
        label = self.label_to_idx.get(emotion_str, -1)
        if label == -1:
            raise ValueError(f"未知表情标签: {emotion_str}")

        total_frames = len(vis_frames)

        # 确定要使用的帧索引及实际长度
        if self.seq_len is not None:
            if total_frames >= self.seq_len:
                start = random.randint(0, total_frames - self.seq_len)
                indices = list(range(start, start + self.seq_len))
            else:
                # 补齐策略：重复最后一帧
                indices = list(range(total_frames)) + [total_frames - 1] * (self.seq_len - total_frames)
            actual_len = self.seq_len
        else:
            indices = list(range(total_frames))
            actual_len = total_frames

        # 为整个序列生成统一的几何变换种子
        geo_seed = random.randint(0, 2**32)

        vis_seq = []
        ir_seq = []

        for i in indices:
            vis_path = os.path.join(self.data_root, vis_frames[i])
            ir_path  = os.path.join(self.data_root, ir_frames[i])

            vis_img = Image.open(vis_path).convert('RGB')
            ir_img  = Image.open(ir_path).convert('L')

            # 同步几何变换
            vis_img, ir_img = self._sync_geometry(vis_img, ir_img, geo_seed)

            # 颜色抖动（仅可见光）
            if self.rgb_color_jitter is not None:
                vis_img = self.rgb_color_jitter(vis_img)

            # 转为张量并归一化
            vis_tensor = self.vis_post(vis_img)
            ir_tensor  = self.ir_post(ir_img)

            vis_seq.append(vis_tensor)
            ir_seq.append(ir_tensor)

        # 堆叠为 (T, C, H, W) 和 (T, 1, H, W)
        vis_seq = torch.stack(vis_seq, dim=0)
        ir_seq  = torch.stack(ir_seq, dim=0)

        return vis_seq, ir_seq, label, actual_len

    @staticmethod
    def collate_fn(batch):
        """
        自定义 batch 合并函数，对变长序列进行填充。
        batch 中每个元素为 (vis_seq, ir_seq, label, length)
        返回：
            vis_padded (B, max_T, C, H, W)
            ir_padded  (B, max_T, 1, H, W)
            labels (B,)
            lengths (B,)  原始长度（不含 padding）
        """
        vis_seqs, ir_seqs, labels, lengths = zip(*batch)

        # 记录每个样本的有效长度
        lengths = torch.tensor(lengths, dtype=torch.long)

        # 填充序列 (假设 T 维在 dim=0，batch_first=True 会将其移到 dim=1)
        vis_padded = pad_sequence(vis_seqs, batch_first=True)   # (B, max_T, C, H, W)
        ir_padded  = pad_sequence(ir_seqs,  batch_first=True)   # (B, max_T, 1, H, W)

        labels = torch.tensor(labels, dtype=torch.long)
        return vis_padded, ir_padded, labels, lengths