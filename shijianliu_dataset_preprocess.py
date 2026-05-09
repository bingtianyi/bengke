#!/usr/bin/env python3
"""
preprocess_temporal_robust_align.py
从原始 Oulu‑CASIA NI/VL 数据集生成旋转校正、对齐裁剪的时序序列。
输出 224x224 分辨率，支持暗光/弱光场景下以红外为基准对齐可见光。
"""

import os, json, re, cv2, numpy as np, mediapipe as mp
from pathlib import Path
from tqdm import tqdm

# ===================== 配置参数 =====================
ORIGINAL_ROOT = r"E:\shujuji\Oulu\Oulu_CASIA_NIR_VIS"
NI_DIR = os.path.join(ORIGINAL_ROOT, "NI")   # 红外图像目录
VL_DIR = os.path.join(ORIGINAL_ROOT, "VL")   # 可见光图像目录

OUTPUT_ROOT = "data/temporal_data"
IMAGE_SIZE = 224            # 输出人脸图像的固定尺寸
MIN_FRAMES = 3              # 一个人至少需要这么多配对帧才保留
TRAIN_COUNT = 64            # 前64人作为训练集
VAL_COUNT = 16              # 后15人作为验证集
LIGHTS = ["Dark", "Weak", "Strong"]
EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

# ---------- MediaPipe 配置 ----------
mp_face_mesh = mp.solutions.face_mesh

# 用于计算眼睛中心的轮廓点索引
LEFT_EYE_IDX  = [33, 133, 157, 158, 159, 160]
RIGHT_EYE_IDX = [362, 263, 387, 388, 389, 390]
# 用于空间对齐的5个特征点：左眼中心、右眼中心、鼻尖、左嘴角、右嘴角
ALIGN_5_POINTS = [33, 263, 1, 61, 291]

# ===================== 工具函数 =====================
def natural_sort_key(name):
    """按数字大小排序文件名，如 '001.jpg' -> [1]"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

def extract_person_number(pid):
    """从 Pxxx 中提取数字"""
    return int(pid[1:]) if pid.startswith('P') else int(pid)

def read_image(path, gray=False):
    """读取图像，可选灰度模式"""
    if gray:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.imread(str(path))

def detect_landmarks(image, face_mesh_instance):
    """检测人脸关键点，返回 (成功标志, landmarks, 图像宽, 图像高)"""
    if image is None:
        return False, None, 0, 0
    h, w = image.shape[:2]
    # 转换到RGB
    if len(image.shape) == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    results = face_mesh_instance.process(rgb)
    if not results.multi_face_landmarks:
        # 尝试直方图均衡化（对红外/灰度图特别有效）
        gray = image if len(image.shape)==2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        rgb_eq = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
        results = face_mesh_instance.process(rgb_eq)
        if not results.multi_face_landmarks:
            return False, None, w, h
    return True, results.multi_face_landmarks[0].landmark, w, h

def get_eye_centers(landmarks, w, h):
    """计算左右眼球中心坐标"""
    left_pts  = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE_IDX], dtype=np.float32)
    right_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE_IDX], dtype=np.float32)
    return left_pts.mean(axis=0), right_pts.mean(axis=0)

def get_5_points(landmarks, w, h):
    """提取5个配准关键点"""
    pts = []
    for idx in ALIGN_5_POINTS:
        pt = landmarks[idx]
        pts.append([pt.x * w, pt.y * h])
    return np.array(pts, dtype=np.float32)

def compute_similarity_transform(src_pts, dst_pts):
    """计算相似变换矩阵（旋转+平移+均匀缩放），使用RANSAC"""
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                        ransacReprojThreshold=5.0)
    return M

def rotate_and_crop(image, eye_centers, out_size=224):
    h, w = image.shape[:2]
    left_eye, right_eye = eye_centers[0], eye_centers[1]

    # 1. 旋转校正
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    # 2. 旋转后的眼睛坐标
    left_homo  = np.array([left_eye[0], left_eye[1], 1.0])
    right_homo = np.array([right_eye[0], right_eye[1], 1.0])
    new_left  = M_rot @ left_homo
    new_right = M_rot @ right_homo

    # 3. 眼距 & 修正后的中心点
    eye_dist = np.linalg.norm(new_right - new_left)
    eye_mid = (new_left + new_right) / 2.0
    cx = eye_mid[0]
    cy = eye_mid[1] + 0.15 * eye_dist   # 中心下移，避免切掉下巴

    # 4. 正方形边长（略大于 2 倍眼距）
    side = int(2.2 * eye_dist)
    x1 = int(cx - side // 2)
    y1 = int(cy - side // 2)

    # 5. 边界约束（确保裁剪区域完全在图像内）
    x1 = max(0, min(x1, w - side))
    y1 = max(0, min(y1, h - side))

    # 6. 裁剪并 resize
    cropped = rotated[y1:y1+side, x1:x1+side]
    resized = cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return resized

def robust_align_and_crop(vis_img, ir_img, face_mesh_inst):
    """
    鲁棒的对齐裁剪函数：自动选择检测成功的模态作为基准，
    利用5点相似变换将另一模态对齐后，统一进行旋转校正和裁剪。
    返回 (vis_out, ir_out) 或 None (如果两者都无法检测到人脸)。
    """
    # ---- 1. 检测两种模态的关键点 ----
    vis_ok, vis_lm, vw, vh = detect_landmarks(vis_img, face_mesh_inst)
    ir_ok,  ir_lm,  iw, ih = detect_landmarks(ir_img,  face_mesh_inst)

    if not vis_ok and not ir_ok:
        return None   # 完全检测不到人脸

    # ---- 2. 确定基准模态和辅模态 ----
    if vis_ok:
        master_img, slave_img = vis_img, ir_img
        master_lm,  slave_lm  = vis_lm,  ir_lm
        mw, mh = vw, vh
        sw, sh = iw, ih
    else:   # 只有红外成功
        master_img, slave_img = ir_img, vis_img
        master_lm,  slave_lm  = ir_lm,  vis_lm
        mw, mh = iw, ih
        sw, sh = vw, vh

    # ---- 3. 将辅模态对齐到主模态空间 ----
    # 提取5个配准点
    master_pts = get_5_points(master_lm, mw, mh)
    if slave_lm is not None:
        slave_pts = get_5_points(slave_lm, sw, sh)
        M_align = compute_similarity_transform(slave_pts, master_pts)
        if M_align is None:
            return None
        # 变换辅图像
        if len(slave_img.shape) == 2:
            slave_warped = cv2.warpAffine(slave_img, M_align, (mw, mh),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            slave_warped = cv2.warpAffine(slave_img, M_align, (mw, mh),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        # 辅模态未检测到人脸，则无法对齐，只能各自独立裁剪（这种情况极少见）
        # 索性直接用主模态的眼睛坐标裁剪主模态，辅模态完整resize
        eye_master = get_eye_centers(master_lm, mw, mh)
        master_cropped = rotate_and_crop(master_img, eye_master, IMAGE_SIZE)
        # 辅模态粗暴resize
        slave_cropped = cv2.resize(slave_img, (IMAGE_SIZE, IMAGE_SIZE))
        # 返回之前要确保顺序是 vis, ir
        if master_img is vis_img:
            return master_cropped, slave_cropped
        else:
            return slave_cropped, master_cropped

    # ---- 4. 获取主模态的眼睛中心 ----
    eye_master = get_eye_centers(master_lm, mw, mh)

    # ---- 5. 对主模态和已对齐的辅模态应用相同的旋转校正+裁剪 ----
    master_cropped = rotate_and_crop(master_img, eye_master, IMAGE_SIZE)
    slave_cropped  = rotate_and_crop(slave_warped, eye_master, IMAGE_SIZE)  # 眼睛坐标复用主模态的

    # ---- 6. 调整回 vis, ir 的顺序 ----
    if master_img is vis_img:
        return master_cropped, slave_cropped
    else:
        return slave_cropped, master_cropped

# ===================== 主处理流程 =====================
def main():
    # 初始化 MediaPipe（在整个处理过程中复用，提高效率）
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.3
    )

    # ----- 扫描所有人员 -----
    print("🔍 扫描人员目录...")
    persons = set()
    for light in LIGHTS:
        light_path = os.path.join(VL_DIR, light)
        if not os.path.isdir(light_path):
            continue
        for pid in os.listdir(light_path):
            pid_path = os.path.join(light_path, pid)
            if os.path.isdir(pid_path):
                persons.add(pid)
    sorted_persons = sorted(persons, key=extract_person_number)
    print(f"发现 {len(sorted_persons)} 人")

    # ----- 划分训练/验证 -----
    train_persons = sorted_persons[:TRAIN_COUNT]
    val_persons   = sorted_persons[-VAL_COUNT:]
    print(f"训练集: {len(train_persons)} 人, 验证集: {len(val_persons)} 人")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    total_seqs = 0
    total_frames = 0
    skip_persons = 0

    # 逐个划分处理
    for split_name, person_list in [("train", train_persons), ("val", val_persons)]:
        print(f"\n🚀 开始处理 {split_name} 集...")
        vis_out_dir = Path(OUTPUT_ROOT) / split_name / "vis"
        ir_out_dir  = Path(OUTPUT_ROOT) / split_name / "ir"
        sequence_index = []

        for person in tqdm(person_list, desc=f"{split_name}"):
            # ----- 收集该人的所有情绪、所有光照的配对帧列表，并确定最小可用帧数 -----
            person_data = {}   # key: (light, emotion) -> (vis_files_list, ir_files_list)
            min_frames_person = None

            for light in LIGHTS:
                for emotion in EMOTIONS:
                    v_emotion_dir = Path(VL_DIR) / light / person / emotion
                    i_emotion_dir = Path(NI_DIR) / light / person / emotion
                    if not v_emotion_dir.exists() or not i_emotion_dir.exists():
                        continue

                    # 获取可见光文件列表并排序
                    v_files = sorted([f for f in v_emotion_dir.iterdir()
                                      if f.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']],
                                     key=lambda x: natural_sort_key(x.name))
                    i_files = sorted([f for f in i_emotion_dir.iterdir()
                                      if f.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']],
                                     key=lambda x: natural_sort_key(x.name))

                    # 可见光与红外按文件名（不含后缀）配对
                    i_dict = {f.stem: f for f in i_files}
                    paired_v, paired_i = [], []
                    for vf in v_files:
                        if vf.stem in i_dict:
                            paired_v.append(vf)
                            paired_i.append(i_dict[vf.stem])

                    if len(paired_v) == 0:
                        continue

                    # 存储
                    person_data[(light, emotion)] = (paired_v, paired_i)
                    if min_frames_person is None or len(paired_v) < min_frames_person:
                        min_frames_person = len(paired_v)

            # 检查是否满足最低要求
            if not person_data or min_frames_person < MIN_FRAMES:
                skip_persons += 1
                continue

            # 确保六种基本情绪都存在（如果缺少，则跳过该人以保证完整性）
            existing_emotions = {emo for (lit, emo) in person_data.keys()}
            if len(existing_emotions) < len(EMOTIONS):
                skip_persons += 1
                continue

            # 统一帧数 N = min_frames_person
            N = min_frames_person

            # ----- 逐情绪、逐光照生成序列 -----
            for (light, emotion), (v_files, i_files) in person_data.items():
                # 取前 N 帧
                v_seq = v_files[:N]
                i_seq = i_files[:N]

                # 输出目录
                out_vis_emotion = vis_out_dir / light / person / emotion
                out_ir_emotion  = ir_out_dir / light / person / emotion
                out_vis_emotion.mkdir(parents=True, exist_ok=True)
                out_ir_emotion.mkdir(parents=True, exist_ok=True)

                out_v_paths = []
                out_i_paths = []

                for idx, (vf_path, irf_path) in enumerate(zip(v_seq, i_seq)):
                    vis_img = read_image(vf_path, gray=False)
                    ir_img  = read_image(irf_path, gray=True)

                    if vis_img is None or ir_img is None:
                        continue

                    # 执行鲁棒对齐与裁剪
                    result = robust_align_and_crop(vis_img, ir_img, face_mesh)
                    if result is None:
                        # 如果对齐裁剪完全失败，则直接使用原图 resize 作为后备
                        def safe_resize(img, size):
                            if img is None: return None
                            return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
                        vis_cropped = safe_resize(vis_img, IMAGE_SIZE)
                        ir_cropped  = safe_resize(ir_img, IMAGE_SIZE)
                        if vis_cropped is None or ir_cropped is None:
                            continue
                    else:
                        vis_cropped, ir_cropped = result

                    # 保存图像
                    frame_name = f"{idx:04d}.jpg"
                    cv2.imwrite(str(out_vis_emotion / frame_name), vis_cropped)
                    cv2.imwrite(str(out_ir_emotion  / frame_name), ir_cropped)

                    # 记录相对路径（用于 JSON 索引）
                    out_v_paths.append(str((out_vis_emotion / frame_name).relative_to(OUTPUT_ROOT).as_posix()))
                    out_i_paths.append(str((out_ir_emotion  / frame_name).relative_to(OUTPUT_ROOT).as_posix()))
                    total_frames += 1

                # 验证序列完整性
                if len(out_v_paths) == N:
                    sequence_index.append({
                        "person": person,
                        "light": light,
                        "emotion": emotion,
                        "vis_frames": out_v_paths,
                        "ir_frames": out_i_paths
                    })
                    total_seqs += 1
                else:
                    print(f"⚠️ 序列不完整: {split_name}/{light}/{person}/{emotion} (仅{len(out_v_paths)}帧)")

        # ----- 保存当前划分的索引 JSON -----
        json_path = Path(OUTPUT_ROOT) / f"{split_name}_sequences.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sequence_index, f, indent=2, ensure_ascii=False)
        print(f"✅ {split_name} 集完成，共 {len(sequence_index)} 条序列 -> {json_path}")

    print("\n" + "="*60)
    print(f"🎉 数据预处理完成！")
    print(f"   总序列数: {total_seqs}")
    print(f"   总帧数:   {total_frames}")
    print(f"   跳过人数: {skip_persons}")
    print(f"   输出目录: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()