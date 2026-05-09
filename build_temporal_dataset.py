#!/usr/bin/env python3
"""
preprocess_temporal_fixed9.py
从原始 Oulu‑CASIA NI/VL 数据集生成固定 9 帧的时序序列。
取每类情绪最后 9 张配对帧（按文件名排序），仅保留六种情绪均满足该数量的受试者。
输出 224x224 分辨率，支持暗光/弱光场景下以红外为基准对齐可见光。
"""

import os, json, re, cv2, numpy as np, mediapipe as mp
from pathlib import Path
from tqdm import tqdm

# ===================== 配置参数 =====================
ORIGINAL_ROOT = r"E:\shujuji\Oulu\Oulu_CASIA_NIR_VIS"
NI_DIR = os.path.join(ORIGINAL_ROOT, "NI")   # 红外图像目录
VL_DIR = os.path.join(ORIGINAL_ROOT, "VL")   # 可见光图像目录

OUTPUT_ROOT = "temporal_data"
IMAGE_SIZE = 224
REQUIRED_FRAMES = 9          # 每种情绪至少需要 9 帧配对图像
TRAIN_COUNT = 65
VAL_COUNT = 15
LIGHTS = ["Dark", "Weak", "Strong"]
EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

# ---------- MediaPipe 配置 ----------
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDX  = [33, 133, 157, 158, 159, 160]
RIGHT_EYE_IDX = [362, 263, 387, 388, 389, 390]
ALIGN_5_POINTS = [33, 263, 1, 61, 291]

# ===================== 工具函数 =====================
def natural_sort_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

def extract_person_number(pid):
    return int(pid[1:]) if pid.startswith('P') else int(pid)

def read_image(path, gray=False):
    if gray:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.imread(str(path))

def detect_landmarks(image, face_mesh_instance):
    if image is None:
        return False, None, 0, 0
    h, w = image.shape[:2]
    if len(image.shape) == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    results = face_mesh_instance.process(rgb)
    if not results.multi_face_landmarks:
        gray = image if len(image.shape)==2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        rgb_eq = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
        results = face_mesh_instance.process(rgb_eq)
        if not results.multi_face_landmarks:
            return False, None, w, h
    return True, results.multi_face_landmarks[0].landmark, w, h

def get_eye_centers(landmarks, w, h):
    left_pts  = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE_IDX], dtype=np.float32)
    right_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE_IDX], dtype=np.float32)
    return left_pts.mean(axis=0), right_pts.mean(axis=0)

def get_5_points(landmarks, w, h):
    pts = []
    for idx in ALIGN_5_POINTS:
        pt = landmarks[idx]
        pts.append([pt.x * w, pt.y * h])
    return np.array(pts, dtype=np.float32)

def compute_similarity_transform(src_pts, dst_pts):
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                        ransacReprojThreshold=5.0)
    return M

def rotate_and_crop(image, eye_centers, out_size=224):
    h, w = image.shape[:2]
    left_eye, right_eye = eye_centers[0], eye_centers[1]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    left_homo  = np.array([left_eye[0], left_eye[1], 1.0])
    right_homo = np.array([right_eye[0], right_eye[1], 1.0])
    new_left  = M_rot @ left_homo
    new_right = M_rot @ right_homo
    eye_dist = np.linalg.norm(new_right - new_left)
    eye_mid = (new_left + new_right) / 2.0
    cx, cy = eye_mid[0], eye_mid[1] + 0.15 * eye_dist  # 中心下移，避免切掉下巴
    side = int(2.2 * eye_dist)
    x1 = int(cx - side // 2)
    y1 = int(cy - side // 2)
    x1 = max(0, min(x1, w - side))
    y1 = max(0, min(y1, h - side))
    cropped = rotated[y1:y1+side, x1:x1+side]
    resized = cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return resized

def robust_align_and_crop(vis_img, ir_img, face_mesh_inst):
    vis_ok, vis_lm, vw, vh = detect_landmarks(vis_img, face_mesh_inst)
    ir_ok,  ir_lm,  iw, ih = detect_landmarks(ir_img,  face_mesh_inst)
    if not vis_ok and not ir_ok:
        return None
    if vis_ok:
        master_img, slave_img = vis_img, ir_img
        master_lm,  slave_lm  = vis_lm,  ir_lm
        mw, mh = vw, vh
        sw, sh = iw, ih
    else:
        master_img, slave_img = ir_img, vis_img
        master_lm,  slave_lm  = ir_lm,  vis_lm
        mw, mh = iw, ih
        sw, sh = vw, vh
    master_pts = get_5_points(master_lm, mw, mh)
    if slave_lm is not None:
        slave_pts = get_5_points(slave_lm, sw, sh)
        M_align = compute_similarity_transform(slave_pts, master_pts)
        if M_align is None:
            return None
        if len(slave_img.shape) == 2:
            slave_warped = cv2.warpAffine(slave_img, M_align, (mw, mh),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            slave_warped = cv2.warpAffine(slave_img, M_align, (mw, mh),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        eye_master = get_eye_centers(master_lm, mw, mh)
        master_cropped = rotate_and_crop(master_img, eye_master, IMAGE_SIZE)
        slave_cropped = cv2.resize(slave_img, (IMAGE_SIZE, IMAGE_SIZE))
        if master_img is vis_img:
            return master_cropped, slave_cropped
        else:
            return slave_cropped, master_cropped
    eye_master = get_eye_centers(master_lm, mw, mh)
    master_cropped = rotate_and_crop(master_img, eye_master, IMAGE_SIZE)
    slave_cropped  = rotate_and_crop(slave_warped, eye_master, IMAGE_SIZE)
    if master_img is vis_img:
        return master_cropped, slave_cropped
    else:
        return slave_cropped, master_cropped

# ===================== 主处理流程 =====================
def main():
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.3
    )
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
    train_persons = sorted_persons[:TRAIN_COUNT]
    val_persons   = sorted_persons[-VAL_COUNT:]
    print(f"训练集: {len(train_persons)} 人, 验证集: {len(val_persons)} 人")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    total_seqs = 0
    total_frames = 0
    skip_persons = 0

    for split_name, person_list in [("train", train_persons), ("val", val_persons)]:
        print(f"\n🚀 开始处理 {split_name} 集...")
        vis_out_dir = Path(OUTPUT_ROOT) / split_name / "vis"
        ir_out_dir  = Path(OUTPUT_ROOT) / split_name / "ir"
        sequence_index = []

        for person in tqdm(person_list, desc=f"{split_name}"):
            # ----- 收集该人的所有情绪、所有光照的配对帧，检查是否每种情绪都有至少9帧 -----
            person_data = {}
            valid_person = True
            for light in LIGHTS:
                for emotion in EMOTIONS:
                    v_emotion_dir = Path(VL_DIR) / light / person / emotion
                    i_emotion_dir = Path(NI_DIR) / light / person / emotion
                    if not v_emotion_dir.exists() or not i_emotion_dir.exists():
                        valid_person = False
                        break
                    v_files = sorted([f for f in v_emotion_dir.iterdir()
                                      if f.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']],
                                     key=lambda x: natural_sort_key(x.name))
                    i_files = sorted([f for f in i_emotion_dir.iterdir()
                                      if f.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']],
                                     key=lambda x: natural_sort_key(x.name))
                    i_dict = {f.stem: f for f in i_files}
                    paired_v, paired_i = [], []
                    for vf in v_files:
                        if vf.stem in i_dict:
                            paired_v.append(vf)
                            paired_i.append(i_dict[vf.stem])
                    if len(paired_v) < REQUIRED_FRAMES:
                        valid_person = False
                        break
                    person_data[(light, emotion)] = (paired_v, paired_i)
                if not valid_person:
                    break
            if not valid_person:
                skip_persons += 1
                continue

            # ----- 逐情绪、逐光照生成固定9帧序列（取最后9帧） -----
            for (light, emotion), (v_files, i_files) in person_data.items():
                # 取最后9帧
                v_seq = v_files[-REQUIRED_FRAMES:]
                i_seq = i_files[-REQUIRED_FRAMES:]
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
                    result = robust_align_and_crop(vis_img, ir_img, face_mesh)
                    if result is None:
                        def safe_resize(img, size):
                            if img is None: return None
                            return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
                        vis_cropped = safe_resize(vis_img, IMAGE_SIZE)
                        ir_cropped  = safe_resize(ir_img, IMAGE_SIZE)
                        if vis_cropped is None or ir_cropped is None:
                            continue
                    else:
                        vis_cropped, ir_cropped = result
                    frame_name = f"{idx:04d}.jpg"
                    cv2.imwrite(str(out_vis_emotion / frame_name), vis_cropped)
                    cv2.imwrite(str(out_ir_emotion  / frame_name), ir_cropped)
                    out_v_paths.append(str((out_vis_emotion / frame_name).relative_to(OUTPUT_ROOT).as_posix()))
                    out_i_paths.append(str((out_ir_emotion  / frame_name).relative_to(OUTPUT_ROOT).as_posix()))
                    total_frames += 1
                if len(out_v_paths) == REQUIRED_FRAMES:
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