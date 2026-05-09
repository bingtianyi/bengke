#
# import pandas as pd
# train_df = pd.read_csv("data/oulu_aug/train.csv")
# val_df = pd.read_csv("data/oulu_aug/val.csv")
# for df, name in [(train_df, "train"), (val_df, "val")]:
#     prefixes = df['vis_path'].apply(lambda x: x.split('/')[-1].split('_')[0])
#     print(name, prefixes.value_counts(normalize=True))
#
# import csv
# from pathlib import Path
#
# CSV_FILES = [
#     "data/oulu_casia_test2/train.csv",
#     "data/oulu_casia_test2/val.csv",
# ]
#
# def fix_csv(input_path):
#     with open(input_path, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         fieldnames = reader.fieldnames
#         rows = list(reader)
#     for row in rows:
#         row['vis_path'] = row['vis_path'].replace('\\', '/')
#         row['ir_path'] = row['ir_path'].replace('\\', '/')
#     with open(input_path, 'w', encoding='utf-8', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)
#     print(f"✅ 已修复: {input_path}")
#
# for f in CSV_FILES:
#     if Path(f).exists():
#         fix_csv(f)
#     else:
#         print(f"⚠️ 文件不存在: {f}")


# import os
# import csv
# from pathlib import Path
#
# # ====================== 配置 ======================
# OUTPUT_ROOT = r"data/oulu_casia_test3"   # 增强数据集根目录
# SPLITS = ["train", "val"]
# LIGHTS = ["Weak", "Strong", "Dark"]
# EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
# # =================================================
#
# def rebuild_csv():
#     for split in SPLITS:
#         vis_root = Path(OUTPUT_ROOT) / split / "vis"
#         ir_root = Path(OUTPUT_ROOT) / split / "ir"
#         rows = []
#
#         for light in LIGHTS:
#             for emotion in EMOTIONS:
#                 vis_dir = vis_root / light / emotion
#                 ir_dir = ir_root / light / emotion
#
#                 if not vis_dir.exists() or not ir_dir.exists():
#                     continue
#
#                 for vis_file in vis_dir.glob("*"):
#                     if vis_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
#                         continue
#
#                     ir_file = ir_dir / vis_file.name
#                     if not ir_file.exists():
#                         continue
#
#                     # 相对路径，使用正斜杠
#                     vis_rel = str(vis_file.relative_to(OUTPUT_ROOT)).replace('\\', '/')
#                     ir_rel = str(ir_file.relative_to(OUTPUT_ROOT)).replace('\\', '/')
#                     rows.append([vis_rel, ir_rel, emotion])
#
#         # 按路径排序（可选）
#         rows.sort(key=lambda x: x[0])
#
#         # 写入 CSV
#         csv_path = Path(OUTPUT_ROOT) / f"{split}.csv"
#         with open(csv_path, 'w', encoding='utf-8', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['vis_path', 'ir_path', 'label'])
#             writer.writerows(rows)
#
#         print(f"✅ 重建 {split}.csv，共 {len(rows)} 对")
#
# if __name__ == "__main__":
#     rebuild_csv()

#生成csv文件，data/oulu_aug_test4
# import os
# import pandas as pd
# from pathlib import Path
#
# # 配置
# #DATA_ROOT = r"/root/autodl-tmp/renlianqingxushibie/data/oulu_aug_test4"
# DATA_ROOT = r"data/oulu_aug_test4"
# EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
# EMOTION_TO_LABEL = {emo: i for i, emo in enumerate(EMOTIONS)}
# SPLITS = ["train", "val"]
# MODALITIES = ["vis", "ir"]
# # 光照类型（根据实际目录名调整，注意大小写）
# LIGHTS = ["Dark", "Strong", "Weak"]  # 如果目录名是 dark/strong/weak 小写，请修改
#
#
# def collect_pairs(split):
#     pairs = []
#     vis_root = Path(DATA_ROOT) / split / "vis"
#     ir_root = Path(DATA_ROOT) / split / "ir"
#     if not vis_root.exists() or not ir_root.exists():
#         print(f"⚠️ {split} 目录不存在，跳过")
#         return pairs
#
#     # 遍历所有光照
#     for light in LIGHTS:
#         vis_light = vis_root / light
#         ir_light = ir_root / light
#         if not vis_light.exists() or not ir_light.exists():
#             continue
#         # 遍历情绪
#         for emotion in EMOTIONS:
#             vis_emo = vis_light / emotion
#             ir_emo = ir_light / emotion
#             if not vis_emo.exists() or not ir_emo.exists():
#                 continue
#             # 获取 vis 目录下的所有图片文件
#             vis_files = list(vis_emo.glob("*.*"))
#             for vis_file in vis_files:
#                 if vis_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
#                     continue
#                 # 对应的红外文件（假设文件名相同）
#                 ir_file = ir_emo / vis_file.name
#                 if ir_file.exists():
#                     label = EMOTION_TO_LABEL[emotion]
#                     pairs.append({
#                         "vis_path": str(vis_file.relative_to(DATA_ROOT)),
#                         "ir_path": str(ir_file.relative_to(DATA_ROOT)),
#                         "label": label,
#                         "split": split,
#                         "light": light,
#                         "emotion": emotion
#                     })
#     return pairs
#
#
# def main():
#     for split in SPLITS:
#         pairs = collect_pairs(split)
#         if not pairs:
#             print(f"❌ {split} 没有找到任何配对图像")
#             continue
#         df = pd.DataFrame(pairs)
#         # 保存 CSV（只保留需要的列）
#         df[['vis_path', 'ir_path', 'label']].to_csv(f"{split}.csv", index=False)
#         print(f"✅ 生成 {split}.csv，共 {len(df)} 条记录")
#         # 可选：打印前几行预览
#         print(df.head())
#
#
# if __name__ == "__main__":
#     main()

import torch

checkpoint_path = 'output/checkpoint_ema_best.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# 1. 打印检查点的所有顶层键，方便调试
print("检查点包含的键:", checkpoint.keys())

# 2. 自动提取模型权重（支持多种格式）
model_state_dict = None

# 常见候选键
candidate_keys = ['state_dict', 'model_state_dict', 'state', 'model']
for key in candidate_keys:
    if key in checkpoint:
        model_state_dict = checkpoint[key]
        print(f"使用键 '{key}' 提取权重")
        break

# 如果上述键都不存在，则假设整个 checkpoint 就是权重字典（跳过顶层包装）
if model_state_dict is None:
    # 检查 checkpoint 本身是否像是权重字典（包含典型的层名称）
    if any('conv' in k or 'bn' in k or 'weight' in k for k in checkpoint.keys()):
        model_state_dict = checkpoint
        print("直接将整个 checkpoint 作为权重字典")
    else:
        raise KeyError("无法从检查点中识别权重字典，请检查打印的键列表并手动指定。")

# 3. 保存为纯净权重文件
output_path = 'pretrained/mobilevitv3_s.pth'
torch.save(model_state_dict, output_path)
print(f"✅ 权重转换成功并保存至: {output_path}")