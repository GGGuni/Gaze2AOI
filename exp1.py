import cv2
import os
import sys
import json
import random
from ultralytics import YOLO
import shutil
import numpy as np
from PIL import Image

import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import mediapipe as mp

from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from tqdm import tqdm
import re

raw_video_dir = "../../../data/exp1_b"
output_dir = "./result_json"
sample_output_dir = './sample'
yolo_model_path = "../../models/yolov8n-face.pt"
REF_DIR = "../../../data/exp1_ref"

os.makedirs(sample_output_dir, exist_ok=True) 
os.makedirs(output_dir, exist_ok=True) 

GPU_ID = 0
FACE_SIMILARITY_THRESHOLD = 0.7

MARGIN = 0.8
margin_x = MARGIN + 0
margin_y = MARGIN + 0.2




SEG_MODEL_ID = "jonathandinu/face-parsing"
HAIR_LABEL = 13
NECK_LABEL = 17
CLOTH_LABEL = 18

AREA2ID = {"forehead": 1, "left_eye": 2, "right_eye": 3, "left_cheek": 4,
           "nose": 5, "right_cheek": 6, "mouth": 7, "chin": 8}
INDEX2AREA = {0: "background", 1: "forehead", 2: "left_eye", 3: "right_eye",
              4: "left_cheek", 5: "nose", 6: "right_cheek", 7: "mouth",
              8: "chin", 9: "hair", 10: "neck"}
HAIR_ID, NECK_ID, BG_ID = 9, 10, 0

HOUGH_DP, HOUGH_MIN_DIST, HOUGH_PARAM1, HOUGH_PARAM2 = 1.2, 30, 50, 30
HOUGH_MINR, HOUGH_MAXR = 25, 30
MEDIAN_BLUR_K = 5

NUM_VIZ_SAMPLES = 5
NUM_PAD_DEMO = 5
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

FACE_AREAS = {
    'forehead': [67, 103, 54, 21, 71, 70, 63, 105, 66, 107, 9, 336, 296, 334, 293, 300, 301, 251, 284, 332, 297, 338, 10, 109],
    'left_eye': [66, 105, 63, 70, 156, 143, 111, 117, 118, 119, 120, 121, 128, 245, 193, 55, 107],
    'right_eye': [336, 285, 417, 465, 357, 350, 349, 348, 347, 346, 340, 372, 383, 300, 293, 334, 296],
    'nose': [9, 107, 55, 193, 245, 114, 217, 198, 209, 129, 203, 98, 97, 2, 326, 327, 423, 358, 429, 420, 437, 343, 465, 417, 285, 336],
    'mouth': [97, 98, 203, 206, 216, 212, 202, 204, 194, 201, 200, 421, 418, 424, 422, 432, 436, 426, 423, 327, 326, 2],
    'chin': [212, 214, 135, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 364, 434, 432, 422, 424, 418, 421, 200, 201, 194, 204, 202],
    'left_cheek': [143, 156, 70, 71, 21, 162, 127, 234, 93, 132, 58, 172, 136, 135, 214, 212, 216, 206, 203, 129, 209, 198, 217, 114, 245, 128, 121, 120, 119, 118, 117, 111],
    'right_cheek': [356, 389, 251, 301, 300, 383, 372, 340, 346, 347, 348, 349, 350, 357, 465, 343, 437, 420, 429, 358, 327, 423, 426, 436, 432, 434, 364, 379, 365, 397, 288, 361, 323, 454],
}
AREA_INDEX = {"left_eye": 2, "right_eye": 3, "nose": 5, "mouth": 7,
              "forehead": 1, "chin": 8, "left_cheek": 4, "right_cheek": 6}



def rename_to_lowercase(folder_path):
    print(f"--- Renaming files to lowercase: {folder_path} ---", flush=True)
    if not os.path.exists(folder_path):
        print("Path does not exist. Skipping.", flush=True)
        print("-" * 30, flush=True)
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        new_filename = filename.lower()
        new_file_path = os.path.join(folder_path, new_filename)
        if filename != new_filename:
            os.rename(file_path, new_file_path)
    print("File renaming completed.", flush=True)
    print("-" * 30, flush=True)


def analyze_videos(folder_path):
    print(f"--- Starting video analysis: {folder_path} ---", flush=True)
    sorted_filenames = sorted(os.listdir(folder_path))
    for filename in sorted_filenames:
        if filename.lower().endswith('.mp4'):
            video_path = os.path.join(folder_path, filename)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[FAILED] Failed to open video: {filename}", flush=True)
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if fps > 0 else 0
            print(f"File Name: {filename}", flush=True)
            print(f"FPS (Frames per second): {fps:.2f}", flush=True)
            print(f"Total Frames: {total_frames}", flush=True)
            print(f"Time per frame: {1 / fps:.4f} seconds", flush=True)
            print(f"Total Duration: {duration_sec:.2f} seconds", flush=True)
            print("-" * 30, flush=True)
            cap.release()


def check_video_pairs(folder_path):
    print(f"--- Comparing frame counts of video pairs: {folder_path} ---", flush=True)
    files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    pairs = {}

    for f in files:
        if "_gaze" in f:
            key = f.replace("_gaze.mp4", "")
            pairs.setdefault(key, {})["gaze"] = f
        elif "_x" in f:
            key = f.replace("_x.mp4", "")
            pairs.setdefault(key, {})["x"] = f

    for key, pair in sorted(pairs.items()):
        if "gaze" not in pair or "x" not in pair:
            print(f"{key} → Incomplete pair (missing file)", flush=True)
            continue

        def get_frame_count(path):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return None
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return total

        path_gaze = os.path.join(folder_path, pair["gaze"])
        path_x = os.path.join(folder_path, pair["x"])

        n_gaze = get_frame_count(path_gaze)
        n_x = get_frame_count(path_x)

        if n_gaze is None or n_x is None:
            print(f"{key} → Failed to open video", flush=True)
        elif n_gaze == n_x:
            print(f"{key} → Frame counts match: {n_gaze} frames", flush=True)
        else:
            print(f"{key} → Mismatch: gaze={n_gaze}, x={n_x}", flush=True)


def hough_center(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, MEDIAN_BLUR_K)
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT,
                               dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
                               param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
                               minRadius=HOUGH_MINR, maxRadius=HOUGH_MAXR)
    if circles is None or circles.shape[1] == 0:
        return None
    circles = np.round(circles[0]).astype(int)
    x, y, r = circles[np.argmax(circles[:, 2])]
    h, w = bgr.shape[:2]
    return int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)), int(r)

def label_text(id_: int) -> str:
    return f"{INDEX2AREA.get(id_, 'unknown')}({id_})"

def put_label(img, label_id, pos=None):
    h, w = img.shape[:2]
    text = f"{INDEX2AREA.get(label_id, 'unknown')}({label_id})"
    scale = max(0.5, min(2.0, min(h, w) / 400.0))
    thick = max(1, int(round(scale * 2)))

    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    text_width, text_height = text_size

    if pos is None:
        pos = (w - text_width - 10, max(24, int(h * 0.08))) 
    else:
        pass
    cv2.putText(img, text, (pos[0] + 1, pos[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thick, cv2.LINE_AA)
    return img

def pad_and_resize(rgb, pad_ratio=0.12, target_short=512):
    h, w = rgb.shape[:2]
    pad = int(round(min(h, w) * pad_ratio))
    rgb_pad = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    h2, w2 = rgb_pad.shape[:2]
    short = min(h2, w2)
    scale = 1.0
    if short < target_short:
        scale = target_short / float(short)
        rgb_pad = cv2.resize(rgb_pad, (int(w2 * scale), int(h2 * scale)), interpolation=cv2.INTER_LINEAR)
    return rgb_pad, pad, scale

def save_padding_demo(face_only_rgb, region_bgr, out_path):
    h, w = face_only_rgb.shape[:2]
    pad = int(round(min(h, w) * 0.12))
    
    orig_bgr = cv2.cvtColor(face_only_rgb, cv2.COLOR_RGB2BGR)
    refl_bgr = cv2.copyMakeBorder(orig_bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)

    target_h = max(orig_bgr.shape[0], refl_bgr.shape[0], region_bgr.shape[0])
    
    def resize_for_hstack(img):
        h_img, w_img = img.shape[:2]
        if h_img != target_h:
            scale = target_h / float(h_img)
            return cv2.resize(img, (int(w_img * scale), target_h), interpolation=cv2.INTER_LINEAR)
        return img

    orig_resized = resize_for_hstack(orig_bgr)
    refl_resized = resize_for_hstack(refl_bgr)
    region_resized = resize_for_hstack(region_bgr)

    panel = np.concatenate((orig_resized, refl_resized, region_resized), axis=1)

    w1 = orig_resized.shape[1]
    w2 = w1 + refl_resized.shape[1]
    
    def put_title(x, text):
        cv2.putText(panel, text, (x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, text, (x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    put_title(0, "Original")
    put_title(w1, "Reflect101 padding")
    put_title(w2, "Result")
    
    cv2.imwrite(out_path, panel)

def get_facemesh_landmarks(face_rgb):
    h_orig, w_orig = face_rgb.shape[:2]

    padded, pad, scale = pad_and_resize(face_rgb, pad_ratio=0.12, target_short=512)
    h_pad, w_pad = padded.shape[:2]
    
    res = face_mesh_main.process(padded)
    
    if res.multi_face_landmarks:
        lms = res.multi_face_landmarks[0].landmark
        pts = []
        for l in lms:
            x_pad = l.x * w_pad
            y_pad = l.y * h_pad
            x_orig = (x_pad - pad) / scale
            y_orig = (y_pad - pad) / scale
            pts.append((int(np.clip(x_orig, 0, w_orig - 1)), int(np.clip(y_orig, 0, h_orig - 1))))
        return pts
    
    res = face_mesh_loose.process(padded)
    if res.multi_face_landmarks:
        lms = res.multi_face_landmarks[0].landmark
        pts = []
        for l in lms:
            x_pad = l.x * w_pad
            y_pad = l.y * h_pad
            x_orig = (x_pad - pad) / scale
            y_orig = (y_pad - pad) / scale
            pts.append((int(np.clip(x_orig, 0, w_orig - 1)), int(np.clip(y_orig, 0, h_orig - 1))))
        return pts
    
    return None

def get_ref_embedding(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding  

def get_embedding(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding  


print('=================================================', flush=True)
rename_to_lowercase(raw_video_dir)
analyze_videos(raw_video_dir)
check_video_pairs(raw_video_dir)


try:
    yolo_model = YOLO(yolo_model_path)
except Exception as e:
    print(f"[Error] Failed to load YOLO: {e}", flush=True)
    sys.exit()

try:
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

    image_processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_ID)
    seg_model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_ID).to(device).eval()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_main = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh_loose = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.2, min_tracking_confidence=0.2)
except Exception as e:
    print(f"Failed to load model: {e}", flush=True)
    exit()

#device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")


model_pack_name = 'buffalo_l'
app = FaceAnalysis(name=model_pack_name)
app.prepare(ctx_id=GPU_ID)


file_list = sorted(os.listdir(raw_video_dir))

video_pairs = {}
for filename in file_list:
    if filename.endswith('.mp4'):
        base_name_parts = filename.replace('.mp4', '').split('_')
        base_name = '_'.join(base_name_parts[:-1])
        
        if base_name not in video_pairs:
            video_pairs[base_name] = {}
        
        if filename.endswith('_gaze.mp4'):
            video_pairs[base_name]['gaze'] = filename
        elif filename.endswith('_x.mp4'):
            video_pairs[base_name]['x'] = filename

print(f"Num of pair images: {len(video_pairs)}", flush=True)
print("-" * 50, flush=True)

ref_dict = {}  
print("Extracting reference embeddings...")
for path in tqdm(sorted(os.listdir(REF_DIR))):
    img_path = os.path.join(REF_DIR, path)
    emb = get_ref_embedding(img_path)
    if emb is not None:
        ref_id = os.path.splitext(path)[0]
        ref_dict[ref_id] = emb

ref_ids = list(ref_dict.keys())
ref_embs = np.stack([ref_dict[k] for k in ref_ids])  

for base_name, files in video_pairs.items():
    if 'gaze' not in files or 'x' not in files:
        print(f"Error in {base_name}' pair", flush=True)
        continue
    
    try:
        num_str = base_name.split('_')[-1].replace('p', '')
        NUM = int(num_str)
    except (IndexError, ValueError):
        print(f"Error: No valid number in {base_name}", flush=True)
        continue
    
    print(f"--- Start processing: {base_name} (P{NUM}) ---", flush=True)

    x_video_path = os.path.join(raw_video_dir, files['x'])
    cap_x = cv2.VideoCapture(x_video_path)
    if not cap_x.isOpened():
        print(f"[Error] Failed to open video: {x_video_path}", flush=True)
        continue
        
    gaze_video_path = os.path.join(raw_video_dir, files['gaze'])
    cap_gaze = cv2.VideoCapture(gaze_video_path)
    if not cap_gaze.isOpened():
        print(f"[Error] Failed to open video: {gaze_video_path}", flush=True)
        cap_x.release()
        continue
    
    total_frames = int(cap_x.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idx = random.sample(range(total_frames), 10)
    # sample_idx = []
    # for i in range(200):
    #     sample_idx.append(i)

    x_label_data = {} 

    
    print(f" -> Processing video: {files['x']}", flush=True)
    print(f" Total number of frames: {total_frames}", flush=True)
    print(f" Sample bbox frame number: {sample_idx}", flush=True)

    final_results = {f"frame_{i}": {"label": BG_ID, "img_id": -1} for i in range(total_frames)}
    
    m = re.search(r'_p(\d+)_', files['x'])
    p_num = int(m.group(1)) if m else None

    pbar = tqdm(total=total_frames, desc="Processing frames")

    frame_idx = 0
    while frame_idx < total_frames:
        # if frame_idx==50: break
        cap_x.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_x = cap_x.read()
        
        if not ret:
            print(f"Error: Can't read frame_{frame_idx} in no_gaze video", flush=True)
            x_label_data[f'frame_{frame_idx}'] = None 
            frame_idx += 1
            continue
        
        h, w, _ = frame_x.shape
        selected_x1, selected_y1, selected_x2, selected_y2 = None, None, None, None
        
        results = yolo_model(frame_x, verbose=False)[0]
        margin_bboxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id != 0:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw = x2 - x1
            bh = y2 - y1

            new_x1 = max(0, x1 - int(bw * margin_x))
            new_y1 = max(0, y1 - int(bh * margin_y * 0.3))
            new_x2 = min(w, x2 + int(bw * margin_x))
            new_y2 = min(h, y2 + int(bh * margin_y))

            margin_bboxes.append([new_x1, new_y1, new_x2, new_y2])

        num_faces = len(margin_bboxes)

        if num_faces == 1:
            selected_x1, selected_y1, selected_x2, selected_y2 = margin_bboxes[0]

        elif num_faces == 2:
            bbox1 = margin_bboxes[0]
            bbox2 = margin_bboxes[1]
            center_y_line = h / 2
            
            is_both_above_center = (bbox1[1] < center_y_line and bbox1[3] < center_y_line and
                                    bbox2[1] < center_y_line and bbox2[3] < center_y_line)
                                    
            if is_both_above_center:
                if bbox1[0] > bbox2[0]:
                    selected_x1, selected_y1, selected_x2, selected_y2 = bbox1
                else:
                    selected_x1, selected_y1, selected_x2, selected_y2 = bbox2
            else:
                if bbox1[1] < bbox2[1]:
                    selected_x1, selected_y1, selected_x2, selected_y2 = bbox1
                else:
                    selected_x1, selected_y1, selected_x2, selected_y2 = bbox2
                    
        elif num_faces >= 3:
            largest_bbox = None
            max_area = 0
            for bbox in margin_bboxes:
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > max_area:
                    max_area = area
                    largest_bbox = bbox
            
            if largest_bbox:
                selected_x1, selected_y1, selected_x2, selected_y2 = largest_bbox

        label_data = None
        if selected_x1 is not None:
            cropped_img_ori = frame_x[max(0, selected_y1):min(h, selected_y2), max(0, selected_x1):min(w, selected_x2)]
            emb = get_embedding(cropped_img_ori)

            cap_gaze.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_gaze = cap_gaze.read()
            h_gaze, w_gaze, _ = frame_gaze.shape
            
            cropped_img_gaze = frame_gaze[max(0, selected_y1):min(h_gaze, selected_y2), max(0, selected_x1):min(w_gaze, selected_x2)]

            with torch.no_grad():
                inputs = image_processor(images=cropped_img_ori, return_tensors="pt").to(device)
                logits = seg_model(**inputs).logits
                up = nn.functional.interpolate(logits, size=cropped_img_ori.shape[:2], mode="bilinear", align_corners=False)
                labels = up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            
            face_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]
            face_mask = np.isin(labels, face_labels)
            face_only_rgb = cv2.bitwise_and(cropped_img_ori, cropped_img_ori, mask=np.uint8(face_mask) * 255)
            # cv2.imwrite(f"debug/face_only_rgb_{frame_idx}.jpg", face_only_rgb)
            c = hough_center(cropped_img_gaze)

            if c:
                if emb is None:
                    matched_id = -1
                else:
                    sims = cosine_similarity([emb], ref_embs)[0]
                    max_sim = np.max(sims)
                    
                    if max_sim < FACE_SIMILARITY_THRESHOLD:
                        matched_id = -1
                    else:
                        best_idx = np.argmax(sims)
                        matched_id = ref_ids[best_idx]

                cx, cy, r = c
                
                labels[labels == CLOTH_LABEL] = 0 
                hair_mask = (labels == HAIR_LABEL)
                neck_mask = (labels == NECK_LABEL)
                
                face_bin = np.uint8(face_mask * 255)
                face_bin = cv2.morphologyEx(face_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
                contours, _ = cv2.findContours(face_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                face_poly = np.zeros_like(face_bin)
                if len(contours) > 0:
                    cv2.drawContours(face_poly, contours, -1, 255, thickness=cv2.FILLED)
                
                h_ori, w_ori = cropped_img_ori.shape[:2]
                
                landmarks_xy = get_facemesh_landmarks(face_rgb=face_only_rgb)
                
                cx_clip = int(np.clip(cx, 0, w_ori - 1))
                cy_clip = int(np.clip(cy, 0, h_ori - 1))
                
                if landmarks_xy:
                    face_mask_mp = np.zeros((h_ori, w_ori), dtype=np.uint8)
                    for name, idxs in FACE_AREAS.items():
                        pts = np.array([landmarks_xy[i] for i in idxs], dtype=np.int32)
                        if len(pts) >= 3:
                            cv2.fillPoly(face_mask_mp, [pts], AREA_INDEX[name])
                    
                    area_idx = int(face_mask_mp[cy_clip, cx_clip])
                    if 1 <= area_idx <= 8:
                        label_id = area_idx
                    else:
                        if hair_mask[cy_clip, cx_clip]:
                            label_id = HAIR_ID
                        elif neck_mask[cy_clip, cx_clip]:
                            label_id = NECK_ID
                        elif face_mask[cy_clip, cx_clip]: 
                            label_id = 1
                        else:
                            label_id = BG_ID
                    
                    mask_color = np.zeros_like(cropped_img_ori)
                    palette = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0),
                               5: (255, 0, 255), 6: (0, 165, 255), 7: (0, 255, 255), 8: (128, 0, 0)}
                    for area_id, color in palette.items():
                        mask_color[face_mask_mp == area_id] = color
                    
                    region_bgr = cv2.addWeighted(cropped_img_gaze, 0.7, cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR), 0.3, 0)
                    cv2.circle(region_bgr, (cx, cy), r, (0, 255, 255), 2)
                    cv2.drawMarker(region_bgr, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)
                    put_label(region_bgr, label_id)
                    # cv2.imwrite(f"final_{frame_idx}.jpg", region_bgr)
                    
                else:
                    if hair_mask[cy_clip, cx_clip]:
                        label_id = HAIR_ID
                    elif neck_mask[cy_clip, cx_clip]:
                        label_id = NECK_ID
                    else:
                        label_id = BG_ID
                    
                    region_bgr = cropped_img_gaze.copy()
                    cv2.circle(region_bgr, (cx, cy), r, (0, 255, 255), 2)
                    cv2.drawMarker(region_bgr, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)
                    put_label(region_bgr, label_id)
                    # cv2.imwrite(f"final_{frame_idx}.jpg", region_bgr)
            else:
                label_id = BG_ID
                region_bgr = cropped_img_gaze.copy()
                put_label(region_bgr, label_id)
                # cv2.imwrite(f"final_{frame_idx}.jpg", region_bgr)
            
            frame_key = f"frame_{frame_idx}"
            final_results[frame_key] = {"label": int(label_id), "img_id": matched_id}

            if frame_idx in sample_idx:
                result = final_results[frame_key]
                text = f"{result['label']}_{result['img_id']}"
                org = (10, 30) 
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                color = (0, 255, 0) 
                thickness = 2

                cv2.putText(cropped_img_gaze, text, org, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
                cv2.imwrite(os.path.join(sample_output_dir, f"frame_{p_num}_{frame_idx}.jpg"), cropped_img_gaze)

        else: 
            label_id = -1
            matched_id = -1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    

    # with open(f"result_json/exp1_p{p_num}.json", "w", encoding="utf-8") as f:
    #     json.dump(final_results, f, ensure_ascii=False, indent=2)

    os.makedirs(output_dir, exist_ok=True) 
    output_path = os.path.join(output_dir, f"exp1_p{p_num}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)


    cap_x.release()
    print(" -> Done processing cropped_ori", flush=True)