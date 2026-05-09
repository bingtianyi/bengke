# #!/usr/bin/env python3
# """
# generate_csv.py
#
# 从已对齐的数据集目录生成 train.csv 和 val.csv
# 每行格式：vis_path, ir_path, label
# """
#
# import csv
# from pathlib import Path
#
# # ================== 请修改以下路径 ==================
# ALIGNED_ROOT = Path("data/oulu_casia_test2")   # 对齐后的根目录
# OUTPUT_DIR = Path("data/oulu_casia_test2")                      # CSV 输出目录（默认为当前目录）
# # =================================================
#
# # 支持的图像后缀
# IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
#
# # Oulu-CASIA 目录结构示例：train/vis/light/person/emotion/image.jpg
# # 我们提取 emotion 作为 label，如需更细粒度可自行调整
# def extract_label(vis_path: Path) -> str:
#     """
#     从 vis 路径中提取类别标签。
#     根据实际目录结构调整，例如 emotion 文件夹名。
#     """
#     # 假设 vis 路径为：.../vis/light/person/emotion/image.jpg
#     # 则取 emotion 部分
#     parts = vis_path.parts
#     # 寻找 vis 后的第 3 级（light/person/emotion）
#     try:
#         vis_idx = parts.index("vis")
#         label = parts[vis_idx + 3]   # 0:light, 1:person, 2:emotion
#         return label
#     except (ValueError, IndexError):
#         # 如果结构不符，尝试取父目录名
#         return vis_path.parent.name
#
#
# def generate_split_csv(split: str, output_csv: Path):
#     split_root = ALIGNED_ROOT / split
#     vis_root = split_root / "vis"
#     ir_root = split_root / "ir"
#
#     if not vis_root.is_dir() or not ir_root.is_dir():
#         print(f"警告：跳过 {split}，目录不存在")
#         return
#
#     rows = []
#     for vis_path in vis_root.rglob("*"):
#         if vis_path.suffix.lower() not in IMAGE_EXTS:
#             continue
#         # 构建 IR 对应路径
#         rel = vis_path.relative_to(vis_root)
#         ir_path = ir_root / rel
#         if not ir_path.is_file():
#             print(f"警告：缺少 IR 对应文件 {ir_path}")
#             continue
#
#         label = extract_label(vis_path)
#         # 写入相对路径（相对于 ALIGNED_ROOT）
#         vis_rel = vis_path.relative_to(ALIGNED_ROOT)
#         ir_rel = ir_path.relative_to(ALIGNED_ROOT)
#         rows.append([str(vis_rel), str(ir_rel), label])
#
#     # 按路径排序
#     rows.sort(key=lambda x: x[0])
#
#     # 写入 CSV
#     with open(output_csv, "w", encoding="utf-8", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["vis_path", "ir_path", "label"])
#         writer.writerows(rows)
#
#     print(f"{split}: 生成 {len(rows)} 条记录，保存至 {output_csv}")
#
#
# if __name__ == "__main__":
#     OUTPUT_DIR.mkdir(exist_ok=True)
#     generate_split_csv("train", OUTPUT_DIR / "train.csv")
#     generate_split_csv("val", OUTPUT_DIR / "val.csv")
#     print("完成！")


import os
import csv
import mediapipe as mp
import cv2
from tqdm import tqdm

# ================= 配置 =================
DATA_ROOT = r"E:\shujuji\FER-2013"        # 数据集根目录
EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
SPLITS = ["train", "val"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# MediaPipe 人脸检测初始化
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def detect_face(image_path):
    """检测图像是否包含人脸，返回 True/False"""
    img = cv2.imread(image_path)
    if img is None:
        return False
    # MediaPipe 需要 RGB 输入
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    return results.detections is not None and len(results.detections) > 0

def generate_cleaned_csv(split):
    """为指定划分生成清洗后的 CSV，仅保留包含人脸的图像"""
    split_dir = os.path.join(DATA_ROOT, split)
    if not os.path.isdir(split_dir):
        print(f"⚠️ 目录不存在：{split_dir}")
        return

    rows = []
    total_files = 0
    kept_files = 0

    for emotion in EMOTIONS:
        emotion_dir = os.path.join(split_dir, emotion)
        if not os.path.isdir(emotion_dir):
            print(f"   ⚠️ 子目录不存在：{emotion_dir}")
            continue

        for filename in os.listdir(emotion_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            file_path = os.path.join(emotion_dir, filename)
            total_files += 1
            if detect_face(file_path):
                # 记录相对路径，格式与之前一致
                rel_path = os.path.join(emotion, filename)
                rows.append((rel_path, emotion))
                kept_files += 1

    if total_files == 0:
        print(f"   ❌ {split} 划分中未找到任何图像")
        return

    # 写入新的 CSV
    csv_path = f"{split}_cleaned.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)

    removed = total_files - kept_files
    print(f"✅ {split} 清洗完成：原有 {total_files} 张，保留 {kept_files} 张，移除 {removed} 张")
    print(f"   新文件保存为：{csv_path}")

# ================= 执行 =================
if __name__ == "__main__":
    for split in SPLITS:
        generate_cleaned_csv(split)

    # 释放 MediaPipe 资源
    face_detection.close()