
#!/usr/bin/env python3
"""
regenerate_csv_with_light.py
重新生成包含光照信息的 train.csv 和 val.csv
适用于无 Person 层级的 Oulu‑CASIA 数据集结构：
    data_root/
        train/
            vis/
                Light/          ← 光照文件夹 (Weak/Strong/Dark)
                    Emotion/    ← 表情类别 (如 Anger)
                        image.jpg
            ir/                 ← 红外图像对应相同结构
        val/
            (同上)
输出的 CSV 格式：vis_path, ir_path, label
所有路径均为相对于 data_root 的路径，使用正斜杠。
"""

import os
import csv
from pathlib import Path

# ====================== 配置（请根据实际情况修改）======================
DATA_ROOT = "E:/pycharm_lianxi/renlianqingxushibie/data/oulu_aug_test4"   # 原始数据集根目录
SPLITS = ["train", "val"]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
# =====================================================================

def get_label_from_path(vis_path: Path, vis_root: Path) -> str:
    """
    从图像路径中提取情感类别（如 Anger, Disgust 等）。
    目录结构：vis_root / Light / Emotion / image.jpg
    返回首字母大写的表情名称，如 'Anger'。
    """
    rel = vis_path.relative_to(vis_root)
    parts = rel.parts   # e.g. ('Dark', 'Anger', '001.jpg')
    if len(parts) >= 2:
        emotion = parts[-2]   # 倒数第二级目录为表情
    else:
        emotion = vis_path.parent.name
    return emotion.capitalize()

def generate_csv(split: str, output_dir: Path):
    vis_root = Path(DATA_ROOT) / split / "vis"
    ir_root = Path(DATA_ROOT) / split / "ir"

    if not vis_root.is_dir():
        print(f"⚠️ 跳过 {split}：可见光目录不存在 ({vis_root})")
        return
    if not ir_root.is_dir():
        print(f"⚠️ 跳过 {split}：红外目录不存在 ({ir_root})")
        return

    rows = []
    for vis_path in vis_root.rglob("*"):
        if vis_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # 构造红外路径：将 /vis/ 替换为 /ir/
        vis_rel = vis_path.relative_to(Path(DATA_ROOT))
        ir_rel = Path(*[part if part != "vis" else "ir" for part in vis_rel.parts])
        ir_path = Path(DATA_ROOT) / ir_rel

        if not ir_path.exists():
            print(f"⚠️ 红外缺失: {ir_rel}")
            continue

        label = get_label_from_path(vis_path, vis_root)
        vis_str = vis_rel.as_posix()
        ir_str = ir_rel.as_posix()
        rows.append((vis_str, ir_str, label))

    rows.sort(key=lambda x: x[0])

    csv_path = output_dir / f"{split}.csv"
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["vis_path", "ir_path", "label"])
        writer.writerows(rows)

    print(f"✅ {split}.csv 已生成，共 {len(rows)} 对样本 (→ {csv_path})")

def main():
    output_dir = Path(DATA_ROOT)   # CSV 直接存放在数据根目录，覆盖原文件
    for split in SPLITS:
        generate_csv(split, output_dir)

if __name__ == "__main__":
    main()