
#图3-1 双目摄像头导致人脸图像空间差异
# import os
# import random
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import mediapipe as mp
#
# # ================== 配置参数 ==================
# DATASET_ROOT = r"E:\pycharm_lianxi\renlianqingxushibie\data\oulu_casia"
# SPLIT = "train"                 # train 或 val
# EMOTION = "Anger"               # 展示的情绪（首字母大写）
# NUM_LIGHTS = 3                  # 三种光照：Dark, Weak, Strong
# OUTPUT_FILE = "disparity_across_lights.png"
#
# # MediaPipe 配置
# mp_face_mesh = mp.solutions.face_mesh
# LEFT_EYE_IDX = 33
# RIGHT_EYE_IDX = 263
#
# random.seed(2026)  # 可复现结果
#
# # ================== 辅助函数 ==================
# def natural_sort_key(s):
#     import re
#     return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
#
# def get_paired_images(vis_root, ir_root, light, person_id, emotion):
#     """给定光照、受试者、情绪，返回一对可见光与红外图像的路径"""
#     vis_dir = os.path.join(vis_root, light, person_id, emotion)
#     ir_dir = os.path.join(ir_root, light, person_id, emotion)
#     if not os.path.isdir(vis_dir) or not os.path.isdir(ir_dir):
#         return None, None
#
#     vis_files = sorted(
#         [f for f in os.listdir(vis_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))],
#         key=natural_sort_key
#     )
#     ir_files = sorted(
#         [f for f in os.listdir(ir_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))],
#         key=natural_sort_key
#     )
#     # 按文件名 stem 匹配第一对
#     ir_stems = {os.path.splitext(f)[0]: f for f in ir_files}
#     for vf in vis_files:
#         stem = os.path.splitext(vf)[0]
#         if stem in ir_stems:
#             return os.path.join(vis_dir, vf), os.path.join(ir_dir, ir_stems[stem])
#     return None, None
#
# def annotate_disparity(vis_img, ir_img):
#     """标注双眼中心并返回偏移文本"""
#     with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
#                                min_detection_confidence=0.5) as face_mesh:
#         # 可见光
#         rgb_vis = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
#         res_vis = face_mesh.process(rgb_vis)
#         vis_lm = res_vis.multi_face_landmarks[0].landmark if res_vis.multi_face_landmarks else None
#
#         # 红外（灰度转三通道）
#         ir_gray = ir_img if len(ir_img.shape)==2 else cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
#         ir_rgb = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2RGB)
#         res_ir = face_mesh.process(ir_rgb)
#         ir_lm = res_ir.multi_face_landmarks[0].landmark if res_ir.multi_face_landmarks else None
#
#     def draw_eyes(img, lm, color):
#         h, w = img.shape[:2]
#         if lm is None:
#             return None, None
#         left_pt = (int(lm[LEFT_EYE_IDX].x * w), int(lm[LEFT_EYE_IDX].y * h))
#         right_pt = (int(lm[RIGHT_EYE_IDX].x * w), int(lm[RIGHT_EYE_IDX].y * h))
#         cv2.circle(img, left_pt, 4, color, -1)
#         cv2.circle(img, right_pt, 4, color, -1)
#         cv2.line(img, left_pt, right_pt, color, 2)
#         return left_pt, right_pt
#
#     vis_copy = vis_img.copy()
#     ir_copy = cv2.cvtColor(ir_img.copy(), cv2.COLOR_GRAY2BGR) if len(ir_img.shape)==2 else ir_img.copy()
#     draw_eyes(vis_copy, vis_lm, (0, 255, 0))
#     draw_eyes(ir_copy, ir_lm, (0, 0, 255))
#
#     disp_text = ""
#     if vis_lm and ir_lm:
#         h, w = vis_img.shape[:2]
#         lv = (int(vis_lm[LEFT_EYE_IDX].x * w), int(vis_lm[LEFT_EYE_IDX].y * h))
#         rv = (int(vis_lm[RIGHT_EYE_IDX].x * w), int(vis_lm[RIGHT_EYE_IDX].y * h))
#         li = (int(ir_lm[LEFT_EYE_IDX].x * w), int(ir_lm[LEFT_EYE_IDX].y * h))
#         ri = (int(ir_lm[RIGHT_EYE_IDX].x * w), int(ir_lm[RIGHT_EYE_IDX].y * h))
#         dx_left = lv[0] - li[0]
#         dy_left = lv[1] - li[1]
#         dx_right = rv[0] - ri[0]
#         dy_right = rv[1] - ri[1]
#         disp_text = f"Left: ({dx_left:+.1f}, {dy_left:+.1f}) px   Right: ({dx_right:+.1f}, {dy_right:+.1f}) px"
#     return vis_copy, ir_copy, disp_text
#
# # ================== 主流程 ==================
# def main():
#     vis_root = os.path.join(DATASET_ROOT, SPLIT, "vis")
#     ir_root = os.path.join(DATASET_ROOT, SPLIT, "ir")
#     if not os.path.isdir(vis_root) or not os.path.isdir(ir_root):
#         print(f"目录不存在：{vis_root} 或 {ir_root}")
#         return
#
#     lights = ["Dark", "Weak", "Strong"]
#     fig, axes = plt.subplots(NUM_LIGHTS, 2, figsize=(10, 4*NUM_LIGHTS))
#     if NUM_LIGHTS == 1:
#         axes = [axes]
#
#     for row, light in enumerate(lights):
#         # 获取该光照下有目标情绪的所有人员
#         light_vis_dir = os.path.join(vis_root, light)
#         if not os.path.isdir(light_vis_dir):
#             axes[row][0].set_title(f"{light} - no data", fontsize=12)
#             axes[row][1].set_title(f"{light} - no data", fontsize=12)
#             axes[row][0].axis('off')
#             axes[row][1].axis('off')
#             continue
#
#         persons = [d for d in os.listdir(light_vis_dir)
#                    if os.path.isdir(os.path.join(light_vis_dir, d))]
#         # 筛选出有该情绪的人员
#         available_persons = []
#         for p in persons:
#             vis_dir = os.path.join(vis_root, light, p, EMOTION)
#             ir_dir = os.path.join(ir_root, light, p, EMOTION)
#             if os.path.isdir(vis_dir) and os.path.isdir(ir_dir):
#                 available_persons.append(p)
#
#         if not available_persons:
#             axes[row][0].set_title(f"{light} - no subject", fontsize=12)
#             axes[row][1].set_title(f"{light} - no subject", fontsize=12)
#             axes[row][0].axis('off')
#             axes[row][1].axis('off')
#             continue
#
#         person = random.choice(available_persons)
#         print(f"{light}: 选择受试者 {person}")
#
#         vis_path, ir_path = get_paired_images(vis_root, ir_root, light, person, EMOTION)
#         if vis_path is None:
#             axes[row][0].set_title(f"{light} - missing pair", fontsize=12)
#             axes[row][1].set_title(f"{light} - missing pair", fontsize=12)
#             axes[row][0].axis('off')
#             axes[row][1].axis('off')
#             continue
#
#         vis_img = cv2.imread(vis_path)
#         ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
#         if vis_img is None or ir_img is None:
#             continue
#
#         vis_annot, ir_annot, disp_text = annotate_disparity(vis_img, ir_img)
#
#         axes[row][0].imshow(cv2.cvtColor(vis_annot, cv2.COLOR_BGR2RGB))
#         axes[row][0].set_title(f"VIS - {light}", fontsize=12)
#         axes[row][0].axis('off')
#
#         axes[row][1].imshow(cv2.cvtColor(ir_annot, cv2.COLOR_BGR2RGB))
#         axes[row][1].set_title(f"IR - {light}\n{disp_text}", fontsize=10)
#         axes[row][1].axis('off')
#
#     plt.suptitle(f"Inter‑modal disparity under different lighting conditions (emotion: {EMOTION})",
#                  fontsize=14, y=1.02)
#     plt.tight_layout()
#     plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
#     print(f"图像已保存为 {OUTPUT_FILE}")
#     plt.show()
#
# if __name__ == "__main__":
#     main()



#图3-2 Oulu-CASIA面部表情数据集示例
# import os
# import random
# import cv2
# import matplotlib.pyplot as plt
#
# # ================== 配置 ==================
# DATASET_ROOT = r"E:\pycharm_lianxi\renlianqingxushibie\data\oulu_casia"
# SPLIT = "train"                 # 使用训练集
# LIGHTS = ["Dark", "Weak", "Strong"]          # 三种光照条件
# ALL_EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
#
# random.seed(2022)               # 可复现结果
#
# OUTPUT_FILE = "oulu_casia_examples_across_lights.png"
#
# # ================== 辅助函数 ==================
# def natural_sort_key(s):
#     import re
#     return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
#
# def get_random_sample_for_emotion(vis_root, ir_root, light, emotion, exclude_persons=None):
#     """在给定光照、情绪下随机选择一个受试者（排除某些人），返回 (person, vis_path, ir_path)"""
#     if exclude_persons is None:
#         exclude_persons = set()
#
#     light_vis_dir = os.path.join(vis_root, light)
#     if not os.path.isdir(light_vis_dir):
#         return None, None, None
#
#     # 列出所有受试者，排除已选中的
#     persons = [d for d in os.listdir(light_vis_dir)
#                if os.path.isdir(os.path.join(light_vis_dir, d)) and d not in exclude_persons]
#     if not persons:
#         return None, None, None
#
#     random.shuffle(persons)
#
#     for person in persons:
#         vis_emotion_dir = os.path.join(light_vis_dir, person, emotion)
#         ir_emotion_dir = os.path.join(ir_root, light, person, emotion)
#         if not os.path.isdir(vis_emotion_dir) or not os.path.isdir(ir_emotion_dir):
#             continue
#
#         # 获取图像文件
#         vis_files = sorted(
#             [f for f in os.listdir(vis_emotion_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))],
#             key=natural_sort_key
#         )
#         ir_files = sorted(
#             [f for f in os.listdir(ir_emotion_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))],
#             key=natural_sort_key
#         )
#
#         # 匹配第一对文件名 stem 相同的图像
#         ir_stems = {os.path.splitext(f)[0]: f for f in ir_files}
#         for vf in vis_files:
#             stem = os.path.splitext(vf)[0]
#             if stem in ir_stems:
#                 vis_path = os.path.join(vis_emotion_dir, vf)
#                 ir_path = os.path.join(ir_emotion_dir, ir_stems[stem])
#                 return person, vis_path, ir_path
#
#     return None, None, None
#
# # ================== 主流程 ==================
# def main():
#     vis_root = os.path.join(DATASET_ROOT, SPLIT, "vis")
#     ir_root = os.path.join(DATASET_ROOT, SPLIT, "ir")
#     if not os.path.isdir(vis_root) or not os.path.isdir(ir_root):
#         print(f"目录不存在：{vis_root} 或 {ir_root}")
#         return
#
#     # 从六种情绪中随机抽取三种不同的情绪，分配给三个光照
#     selected_emotions = random.sample(ALL_EMOTIONS, 3)
#
#     samples = []
#     used_persons = set()   # 避免同一个人被重复选中（但不同光照本身人员是可以重复的，这里不强求）
#     for light, emotion in zip(LIGHTS, selected_emotions):
#         person, vis_path, ir_path = get_random_sample_for_emotion(
#             vis_root, ir_root, light, emotion
#         )
#         if vis_path is None:
#             print(f"警告：无法在 {light} 光照下为情绪 {emotion} 找到样本，跳过")
#             continue
#         samples.append((light, person, emotion, vis_path, ir_path))
#         used_persons.add(person)   # 记录已用人员，但这里没有跨光照限制
#
#     if len(samples) == 0:
#         print("没有找到任何样本，请检查数据集路径和结构。")
#         return
#
#     num = len(samples)
#     fig, axes = plt.subplots(num, 2, figsize=(8, 3.5 * num))
#     if num == 1:
#         axes = [axes]
#
#     for i, (light, person, emotion, vis_path, ir_path) in enumerate(samples):
#         vis_img = cv2.imread(vis_path)
#         ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
#         if vis_img is None or ir_img is None:
#             continue
#
#         axes[i][0].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
#         axes[i][0].set_title(f"VIS - {light} - {person} - {emotion}", fontsize=11)
#         axes[i][0].axis('off')
#
#         axes[i][1].imshow(ir_img, cmap='gray')
#         axes[i][1].set_title(f"IR - {light} - {person} - {emotion}", fontsize=11)
#         axes[i][1].axis('off')
#
#     plt.suptitle("Oulu‑CASIA 面部表情数据集示例（不同光照、不同情绪）", fontsize=14, y=1.02)
#     plt.tight_layout()
#     plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
#     print(f"图像已保存为 {OUTPUT_FILE}")
#     plt.show()
#
# if __name__ == "__main__":
#     main()

#图3-3 人脸对齐效果图
# import cv2
# import matplotlib.pyplot as plt
# import os
#
# # ================= 配置路径 =================
# # 原始数据集根目录（包含 VL 和 NI 文件夹）
# ORIGINAL_ROOT = r"E:\pycharm_lianxi\renlianqingxushibie\data\oulu_casia"
#
# ALIGNED_ROOT = r"E:\pycharm_lianxi\renlianqingxushibie\data\oulu_casia_test3"
#
# LIGHT = "Strong"
# PERSON = "P002"
# EMOTION = "Disgust"
#
# # ================= 辅助函数 =================
# def natural_sort_key(s):
#     import re
#     return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
#
# def get_first_image(path_dir):
#     if not os.path.isdir(path_dir):
#         return None, None
#     files = sorted([f for f in os.listdir(path_dir)
#                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))],
#                    key=natural_sort_key)
#     if files:
#         return files[0], os.path.join(path_dir, files[0])
#     return None, None
#
# # ================= 读取图像 =================
# orig_vis_dir = os.path.join(ORIGINAL_ROOT, "train", "vis", LIGHT, PERSON, EMOTION)
# orig_ir_dir = os.path.join(ORIGINAL_ROOT, "train", "ir", LIGHT, PERSON, EMOTION)
# aligned_vis_dir = os.path.join(ALIGNED_ROOT, "train", "vis", LIGHT, PERSON, EMOTION)
# aligned_ir_dir = os.path.join(ALIGNED_ROOT, "train", "ir", LIGHT, PERSON, EMOTION)
#
# _, orig_vis_path = get_first_image(orig_vis_dir)
# _, orig_ir_path = get_first_image(orig_ir_dir)
# _, aligned_vis_path = get_first_image(aligned_vis_dir)
# _, aligned_ir_path = get_first_image(aligned_ir_dir)
#
# if not all([orig_vis_path, orig_ir_path, aligned_vis_path, aligned_ir_path]):
#     print("缺少图像文件，请检查路径。")
#     exit()
#
# # 原始图像
# orig_vis = cv2.imread(orig_vis_path)
# orig_ir = cv2.imread(orig_ir_path, cv2.IMREAD_GRAYSCALE)
#
# # 对齐后图像
# aligned_vis = cv2.imread(aligned_vis_path)
# aligned_ir = cv2.imread(aligned_ir_path, cv2.IMREAD_GRAYSCALE)
#
# # ================= 统一高度：将对齐后图像缩放到原始图像高度 =================
# base_h = orig_vis.shape[0]  # 原始图像高度 240
# aligned_vis = cv2.resize(aligned_vis,
#                          (int(aligned_vis.shape[1] * base_h / aligned_vis.shape[0]), base_h))
# aligned_ir = cv2.resize(aligned_ir,
#                         (int(aligned_ir.shape[1] * base_h / aligned_ir.shape[0]), base_h))
#
# # 转换为 RGB 用于显示
# orig_vis_rgb = cv2.cvtColor(orig_vis, cv2.COLOR_BGR2RGB)
# aligned_vis_rgb = cv2.cvtColor(aligned_vis, cv2.COLOR_BGR2RGB)
#
# # ================= 绘制 2×2 对比图（上：原始，下：对齐后） =================
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))
#
# # 第一行：原始
# axes[0, 0].imshow(orig_vis_rgb)
# axes[0, 0].set_title("Original VIS (320×240)", fontsize=12)
# axes[0, 0].axis('off')
#
# axes[0, 1].imshow(orig_ir, cmap='gray')
# axes[0, 1].set_title("Original IR (320×240)", fontsize=12)
# axes[0, 1].axis('off')
#
# # 第二行：对齐后
# axes[1, 0].imshow(aligned_vis_rgb)
# axes[1, 0].set_title(f"Aligned VIS (224×224)", fontsize=12)
# axes[1, 0].axis('off')
#
# axes[1, 1].imshow(aligned_ir, cmap='gray')
# axes[1, 1].set_title(f"Aligned IR (224×224)", fontsize=12)
# axes[1, 1].axis('off')
#
# plt.suptitle(f"Face Alignment Comparison ({LIGHT} - {PERSON} - {EMOTION})", fontsize=14, y=1.02)
# plt.tight_layout()
#
# # 保存图像
# output_path = "face_alignment_comparison.png"
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# print(f"对比图已保存至: {output_path}")
# plt.show()


# #图4-1 双主干多模态融合网络整体架构
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.lines import Line2D
#
# # 设置画布
# fig, ax = plt.subplots(1, 1, figsize=(12, 6))
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 6)
# ax.axis('off')
#
# # 颜色定义
# c_vis = '#4C72B0'      # 可见光分支
# c_ir = '#DD8452'       # 红外分支
# c_fusion = '#55A868'   # 融合模块
# c_head = '#C44E52'     # 分类头
#
# # ================== 绘制输入 ==================
# ax.text(1.0, 4.8, 'Visible Image\n(224×224×3)', ha='center', va='center',
#         fontsize=10, bbox=dict(boxstyle='round', facecolor='white', edgecolor=c_vis, lw=2))
# ax.text(1.0, 1.2, 'IR Image\n(224×224×1)', ha='center', va='center',
#         fontsize=10, bbox=dict(boxstyle='round', facecolor='white', edgecolor=c_ir, lw=2))
#
# # ================== 绘制主干 ==================
# # 可见光主干 (ResNet50)
# ax.text(3.5, 4.8, 'ResNet-50\n(Backbone)', ha='center', va='center',
#         fontsize=10, bbox=dict(boxstyle='round', facecolor=c_vis, alpha=0.7, edgecolor='black'))
# ax.annotate('Feature: 2048-d', xy=(3.5, 4.2), ha='center', fontsize=9, color=c_vis)
#
# # 红外主干 (ConvNeXtV2-Tiny)
# ax.text(3.5, 1.2, 'ConvNeXtV2-Tiny\n(Backbone)', ha='center', va='center',
#         fontsize=10, bbox=dict(boxstyle='round', facecolor=c_ir, alpha=0.7, edgecolor='black'))
# ax.annotate('Feature: 768-d', xy=(3.5, 0.6), ha='center', fontsize=9, color=c_ir)
#
# # 红外适配层
# ax.text(2.2, 1.2, 'IR Adapter\n(1→3 Conv)', ha='center', va='center',
#         fontsize=9, bbox=dict(boxstyle='round', facecolor='white', edgecolor=c_ir, lw=1, linestyle='--'))
#
# # ================== 绘制融合模块 ==================
# fusion_rect = mpatches.FancyBboxPatch((5.5, 1.8), 2.0, 2.4,
#                                       boxstyle="round,pad=0.2",
#                                       facecolor=c_fusion, alpha=0.7, edgecolor='black', lw=2)
# ax.add_patch(fusion_rect)
# ax.text(6.5, 3.8, 'Cross-Modal\nAttention\nFusion', ha='center', va='center',
#         fontsize=11, fontweight='bold', color='white')
# ax.text(6.5, 2.6, 'CFIM + DFIM\nSelf-Attention\nProjection 128-d', ha='center', va='center',
#         fontsize=9, color='white')
#
# # ================== 绘制分类头 ==================
# ax.text(8.5, 3.0, 'Fusion Head\n(128→64→6)', ha='center', va='center',
#         fontsize=10, bbox=dict(boxstyle='round', facecolor=c_head, alpha=0.7, edgecolor='black'))
# ax.text(8.5, 2.2, 'Emotion\nPrediction', ha='center', va='center',
#         fontsize=11, fontweight='bold', color=c_head)
#
# # ================== 绘制箭头 ==================
# # VIS 输入 -> 主干
# ax.annotate('', xy=(2.6, 4.8), xytext=(1.8, 4.8),
#             arrowprops=dict(arrowstyle='->', color=c_vis, lw=2))
# # IR 输入 -> 适配层 -> 主干
# ax.annotate('', xy=(2.6, 1.2), xytext=(1.8, 1.2),
#             arrowprops=dict(arrowstyle='->', color=c_ir, lw=2))
# ax.annotate('', xy=(3.0, 1.2), xytext=(2.6, 1.2),
#             arrowprops=dict(arrowstyle='->', color=c_ir, lw=1))
#
# # 主干 -> 融合
# ax.annotate('', xy=(5.4, 3.4), xytext=(4.2, 4.5),
#             arrowprops=dict(arrowstyle='->', color=c_vis, lw=2, connectionstyle='arc3,rad=0.2'))
# ax.annotate('', xy=(5.4, 2.6), xytext=(4.2, 1.5),
#             arrowprops=dict(arrowstyle='->', color=c_ir, lw=2, connectionstyle='arc3,rad=-0.2'))
#
# # 融合 -> 分类头
# ax.annotate('', xy=(8.0, 3.2), xytext=(7.6, 3.0),
#             arrowprops=dict(arrowstyle='->', color=c_head, lw=2))
#
# # ================== 图例 ==================
# legend_elements = [mpatches.Patch(color=c_vis, alpha=0.7, label='Visible Branch'),
#                    mpatches.Patch(color=c_ir, alpha=0.7, label='Infrared Branch'),
#                    mpatches.Patch(color=c_fusion, alpha=0.7, label='Fusion Module'),
#                    mpatches.Patch(color=c_head, alpha=0.7, label='Classification Head')]
# ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
#
# # ================== 标签 ==================
# ax.set_title('Dual-Backbone Multimodal Expression Recognition Network', fontsize=14, fontweight='bold', pad=15)
#
# plt.tight_layout()
# plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight')
# plt.show()


# #图4-2 跨模态注意力融合模块结构
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
#
# fig, ax = plt.subplots(figsize=(14, 7))
# ax.set_xlim(0, 14)
# ax.set_ylim(0, 7)
# ax.axis('off')
#
# # 颜色定义
# c_vis = '#4C72B0'
# c_ir = '#DD8452'
# c_cfim = '#55A868'
# c_dfim = '#8B5CF6'
# c_mlp = '#E67E22'
# c_attn = '#C44E52'
# c_fused = '#2C3E50'
#
# # ---------- 输入 ----------
# ax.text(1.0, 5.5, '$F_{vis}$\n(2048-d)', ha='center', va='center', fontsize=11,
#         bbox=dict(boxstyle='round', facecolor=c_vis, alpha=0.8, edgecolor='black'))
# ax.text(1.0, 1.5, '$F_{ir}$\n(768-d)', ha='center', va='center', fontsize=11,
#         bbox=dict(boxstyle='round', facecolor=c_ir, alpha=0.8, edgecolor='black'))
#
# # ---------- 投影层 ----------
# ax.text(2.8, 5.5, 'Linear\nProjection', ha='center', va='center', fontsize=9,
#         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', lw=1.5))
# ax.text(2.8, 1.5, 'Linear\nProjection', ha='center', va='center', fontsize=9,
#         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', lw=1.5))
# ax.text(3.7, 6.2, '$p_{vis}$ (128-d)', ha='center', fontsize=9, color=c_vis)
# ax.text(3.7, 0.8, '$p_{ir}$ (128-d)', ha='center', fontsize=9, color=c_ir)
#
# # ---------- 箭头：输入 -> 投影 ----------
# ax.annotate('', xy=(2.2, 5.5), xytext=(1.6, 5.5),
#             arrowprops=dict(arrowstyle='->', color=c_vis, lw=2))
# ax.annotate('', xy=(2.2, 1.5), xytext=(1.6, 1.5),
#             arrowprops=dict(arrowstyle='->', color=c_ir, lw=2))
#
# # ---------- CFIM ----------
# cfim_rect = mpatches.FancyBboxPatch((4.5, 3.2), 1.8, 1.6, boxstyle="round",
#                                      facecolor=c_cfim, alpha=0.6, edgecolor='black', lw=2)
# ax.add_patch(cfim_rect)
# ax.text(5.4, 4.2, 'CFIM', ha='center', fontsize=12, fontweight='bold', color='white')
# ax.text(5.4, 3.8, 'Common Feature\nInjection', ha='center', fontsize=8, color='white')
#
# # ---------- DFIM ----------
# dfim_rect = mpatches.FancyBboxPatch((4.5, 0.8), 1.8, 1.6, boxstyle="round",
#                                      facecolor=c_dfim, alpha=0.6, edgecolor='black', lw=2)
# ax.add_patch(dfim_rect)
# ax.text(5.4, 1.8, 'DFIM', ha='center', fontsize=12, fontweight='bold', color='white')
# ax.text(5.4, 1.4, 'Different Feature\nInjection', ha='center', fontsize=8, color='white')
#
# # 箭头：投影 -> CFIM/DFIM
# ax.annotate('', xy=(4.4, 4.0), xytext=(3.4, 5.2),
#             arrowprops=dict(arrowstyle='->', color=c_cfim, lw=2, connectionstyle='arc3,rad=-0.2'))
# ax.annotate('', xy=(4.4, 1.8), xytext=(3.4, 1.8),
#             arrowprops=dict(arrowstyle='->', color=c_ir, lw=2))
# ax.annotate('', xy=(4.4, 3.8), xytext=(3.4, 5.0),
#             arrowprops=dict(arrowstyle='->', color=c_vis, lw=2, connectionstyle='arc3,rad=0.2'))
# ax.annotate('', xy=(4.4, 1.0), xytext=(3.4, 1.0),
#             arrowprops=dict(arrowstyle='->', color=c_ir, lw=2))
#
# # ---------- 拼接操作 ----------
# ax.text(7.2, 3.5, 'Concatenate\n4×128 = 512-d', ha='center', va='center', fontsize=10,
#         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', lw=2))
#
# # 箭头：CFIM/DFIM -> 拼接
# ax.annotate('', xy=(6.8, 3.6), xytext=(6.3, 4.0),
#             arrowprops=dict(arrowstyle='->', color=c_cfim, lw=1.5))
# ax.annotate('', xy=(6.8, 3.4), xytext=(6.3, 1.6),
#             arrowprops=dict(arrowstyle='->', color=c_dfim, lw=1.5))
#
# # ---------- 融合MLP ----------
# ax.text(9.0, 3.5, 'Fusion\nMLP\n(128-d)', ha='center', va='center', fontsize=10,
#         bbox=dict(boxstyle='round', facecolor=c_mlp, alpha=0.8, edgecolor='black', lw=2))
# ax.text(9.0, 5.0, '$f_{mlp}$', ha='center', fontsize=9, color=c_mlp)
#
# # 箭头：拼接 -> MLP
# ax.annotate('', xy=(8.3, 3.5), xytext=(7.8, 3.5),
#             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
#
# # ---------- 自注意力 ----------
# ax.text(11.0, 3.5, 'Self-Attention\n(4 heads)', ha='center', va='center', fontsize=10,
#         bbox=dict(boxstyle='round', facecolor=c_attn, alpha=0.8, edgecolor='black', lw=2))
# ax.text(11.0, 5.0, '$f_{attn}$', ha='center', fontsize=9, color=c_attn)
#
# # 箭头：MLP -> 自注意力
# ax.annotate('', xy=(10.3, 3.5), xytext=(9.7, 3.5),
#             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
#
# # ---------- 输出特征 ----------
# ax.text(12.8, 3.5, '$F_{fused}$\n(128-d)', ha='center', va='center', fontsize=11,
#         bbox=dict(boxstyle='round', facecolor=c_fused, alpha=0.8, edgecolor='black', lw=2))
#
# # 箭头：自注意力 -> 输出
# ax.annotate('', xy=(12.1, 3.5), xytext=(11.7, 3.5),
#             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
#
# # ---------- 标注 ----------
# # 虚线框：整体模块
# rect = mpatches.FancyBboxPatch((3.8, 0.2), 9.8, 6.6, boxstyle="round,pad=0.3",
#                                  facecolor='none', edgecolor='gray', lw=2, linestyle='--')
# ax.add_patch(rect)
# ax.text(4.0, 6.5, 'Cross-Modal Attention Fusion Module', fontsize=12, fontweight='bold', color='gray')
#
# # 图例
# legend_elements = [mpatches.Patch(color=c_cfim, alpha=0.6, label='CFIM'),
#                    mpatches.Patch(color=c_dfim, alpha=0.6, label='DFIM'),
#                    mpatches.Patch(color=c_mlp, alpha=0.8, label='Fusion MLP'),
#                    mpatches.Patch(color=c_attn, alpha=0.8, label='Self-Attention')]
# ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
#
# plt.tight_layout()
# plt.savefig('CMAF_module.png', dpi=300, bbox_inches='tight')
# plt.show()



#图6-1 系统整体架构图
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# -------------------- 解决中文显示问题 --------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示异常

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# 颜色定义
c_front = '#4C72B0'         # 前端
c_back = '#55A868'          # 后端
c_align = '#8B5CF6'         # 人脸对齐
c_model = '#E67E22'         # 模型管理
c_data = '#DD8452'          # 数据/文件

# ---------- 前端层 ----------
ax.text(1.5, 7.0, 'Web 浏览器 (HTML5)', ha='center', va='center',
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=c_front, alpha=0.8, edgecolor='black'))
ax.text(1.5, 6.0, '图片上传 / 模式选择\n单模态 / 双模态 / 快速 / 专家',
        ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=c_front))

# ---------- 后端 Flask 服务 ----------
ax.text(5.0, 7.0, 'Flask 应用服务器', ha='center', va='center',
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=c_back, alpha=0.8, edgecolor='black'))
ax.text(5.0, 6.0, '路由处理 (/predict_image)\n请求解析、文件保存、调用逻辑',
        ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=c_back))

# ---------- 人脸对齐模块 ----------
ax.text(8.5, 7.0, '人脸对齐模块\n(MediaPipe + OpenCV)', ha='center', va='center',
        fontsize=10, bbox=dict(boxstyle='round', facecolor=c_align, alpha=0.7, edgecolor='black'))
ax.text(8.5, 6.0, '关键点检测 → 旋转校正 → 比例裁剪 → 224×224',
        ha='center', va='center', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=c_align))

# ---------- 模型管理层 ----------
ax.text(5.0, 4.0, '模型管理', ha='center', va='center',
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=c_model, alpha=0.8, edgecolor='black'))

# 四个模型
ax.text(2.5, 3.0, '快速模式\n(双 MobileViTv3‑S)\nmodel_1 + best_model_1.pth',
        ha='center', va='center', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=c_model))
ax.text(5.0, 2.5, '专家模式\n(ResNet50+ConvNeXtV2)\nmodel_2 + best_model_2.pth',
        ha='center', va='center', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=c_model))
ax.text(7.5, 3.0, '时序模型\n(双流+LSTM)\ntemporal_model + best_temporal.pth',
        ha='center', va='center', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=c_model))
# 单模态分支依靠双模态模型的备用头，不单独加载其他模型

# ---------- 数据存储 ----------
ax.text(11.5, 4.0, '临时文件存储\n(uploads/)', ha='center', va='center',
        fontsize=10, bbox=dict(boxstyle='round', facecolor=c_data, alpha=0.7, edgecolor='black'))

# ---------- 箭头与连接 ----------
# 前端 -> Flask
ax.annotate('', xy=(3.8, 6.5), xytext=(2.3, 6.5),
            arrowprops=dict(arrowstyle='->', color=c_front, lw=2))
# Flask -> 人脸对齐
ax.annotate('', xy=(7.5, 6.5), xytext=(5.8, 6.5),
            arrowprops=dict(arrowstyle='->', color=c_back, lw=2))
# 人脸对齐 -> Flask（返回处理后图像）
ax.annotate('', xy=(6.0, 5.8), xytext=(8.0, 5.8),
            arrowprops=dict(arrowstyle='->', color=c_align, lw=1.5, connectionstyle='arc3,rad=0.2'))
# Flask -> 模型管理
ax.annotate('', xy=(5.0, 4.8), xytext=(5.0, 6.0),
            arrowprops=dict(arrowstyle='->', color=c_back, lw=2))
# 模型管理 -> 各模型（内部）
# 单/双模态判断在路由中完成，直接调用对应函数，不单独画线

# 结果返回
ax.annotate('', xy=(2.0, 6.0), xytext=(2.0, 4.5),
            arrowprops=dict(arrowstyle='->', color=c_front, lw=1.5, connectionstyle='arc3,rad=-0.3'))

# ---------- 图例 ----------
legend_elements = [mpatches.Patch(color=c_front, alpha=0.7, label='前端层'),
                   mpatches.Patch(color=c_back, alpha=0.7, label='后端服务'),
                   mpatches.Patch(color=c_align, alpha=0.7, label='人脸对齐'),
                   mpatches.Patch(color=c_model, alpha=0.7, label='模型管理'),
                   mpatches.Patch(color=c_data, alpha=0.7, label='数据存储')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

ax.set_title('双模态表情识别展示系统整体架构', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
plt.show()