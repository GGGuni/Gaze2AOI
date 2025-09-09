import cv2
import os
import shutil
import json
import random
from ultralytics import YOLO
import re
import numpy as np
import random
from pathlib import Path
from PIL import Image

import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from tqdm import tqdm

GPU_ID = 0
NUM = 2
MARGIN = 0.8
raw_video_dir = "../../../data/exp2"
REF_DIR = "../../../data/exp2_ref"
output_dir = "./result_json"
sample_output_dir = "./sample"
model_path = "../../models/yolov8n-face.pt"

os.makedirs(sample_output_dir, exist_ok=True) 
os.makedirs(output_dir, exist_ok=True) 


device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
use_autocast = (device.type == "cuda")

SEG_MODEL_ID = "jonathandinu/face-parsing"
BG_ID = 0
FACE_LABEL = 1
NECK_LABEL = 17
CLOTH_LABEL = 18
REMOVE_LABELS = [CLOTH_LABEL]

NUM_VIZ_SAMPLES = 5
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# device = "cuda:1" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
# use_autocast = (device == "cuda")

DP, MIN_DIST, PARAM1, PARAM2, MIN_R, MAX_R = 1.2, 30, 50, 30, 25, 30
MEDIAN_BLUR_K = 5

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

def trim_below_lowest_neck(labels):
    ys, _ = np.where(labels == NECK_LABEL)
    if ys.size == 0:
        return labels
    y_cut = int(ys.max())
    if y_cut + 1 < labels.shape[0]:
        labels[y_cut + 1:, :] = 0
    return labels

def to_binary_mask(labels):
    return (labels > 0).astype(np.uint8) * 255

def find_gaze_point(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, MEDIAN_BLUR_K)
    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT,
        dp=DP, minDist=MIN_DIST, param1=PARAM1, param2=PARAM2,
        minRadius=MIN_R, maxRadius=MAX_R
    )
    if circles is not None and circles.shape[1] > 0:
        circles = np.round(circles[0]).astype(int)
        x, y, r = circles[np.argmax(circles[:, 2])]
        return [int(x), int(y)]
    return None

def run_segformer(processor, model_segformer, img):
    H, W = img.shape[:2]
    mid = W // 2
    left  = img[:, :mid]
    right = img[:, mid:]

    with torch.no_grad():
        inputs = processor(images=[left, right], return_tensors="pt").to(device)
        logits = model_segformer(**inputs).logits  # (2, C, h', w')
        up_l = F.interpolate(logits[0:1], size=(H, left.shape[1]),  mode="bilinear", align_corners=False)
        up_r = F.interpolate(logits[1:2], size=(H, right.shape[1]), mode="bilinear", align_corners=False)
        lab_l = up_l.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        lab_r = up_r.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    lab_l[lab_l == CLOTH_LABEL] = 0
    lab_r[lab_r == CLOTH_LABEL] = 0

    final_mask = np.zeros((H, W), dtype=np.uint8)
    final_mask[:, :mid][lab_l != 0] = 1 
    final_mask[:, mid:][lab_r != 0] = 2  

    return final_mask

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

def run_arcface(img, ref_embs):
    emb = get_embedding(img)
    sims = cosine_similarity([emb], ref_embs)[0]
    best_idx = np.argmax(sims)
    matched_id = ref_ids[best_idx]

    return matched_id

print('=================================================', flush=True)
rename_to_lowercase(raw_video_dir)
analyze_videos(raw_video_dir)
check_video_pairs(raw_video_dir)

model_yolo = YOLO(model_path)
processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_ID)
model_segformer = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_ID).to(device).eval()

model_pack_name = 'buffalo_l'
app = FaceAnalysis(name=model_pack_name)
app.prepare(ctx_id=0)

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

for base_name, files in video_pairs.items():
    if 'gaze' not in files or 'x' not in files:
        print(f"Error in '{base_name}' pair", flush=True)
        continue
    
    num_str = base_name.split('_')[-1].replace('p', '')
    if not num_str.isdigit():
        print(f"Error: No valid number in {base_name}", flush=True)
        continue
    NUM = int(num_str)
    
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
        continue

    m = re.search(r'_p(\d+)_', files['x'])
    p_num = int(m.group(1)) if m else None

    total_frames = int(cap_x.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idx = random.sample(range(total_frames), 10)
    # sample_idx = []
    # for i in range(200):
    #     sample_idx.append(i)

    final_results = {} 
    
    final_results = {f"frame_{i}": {"label": BG_ID, "img_id": -1} for i in range(total_frames)}
    
    frame_idx = 0

    print(f" -> Processing video: {files['x']}", flush=True)
    print(f" Total number of frames: {total_frames}", flush=True)

    pbar = tqdm(total=total_frames, desc="Processing frames")

    frame_idx = 0
    while frame_idx < total_frames:
        # if frame_idx==100: break
        cap_x.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_x = cap_x.read()

        cap_gaze.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_gaze, frame_gaze = cap_gaze.read()
        h_gaze, w_gaze, _ = frame_gaze.shape
        
        h, w, _ = frame_x.shape
        u_x1, u_y1, u_x2, u_y2 = None, None, None, None
        
        results = model_yolo(frame_x, verbose=False)[0]
        margin_bboxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id != 0:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw = x2 - x1
            bh = y2 - y1
            margin_x = MARGIN + 0
            margin_y = MARGIN + 0.2
            new_x1 = max(0, x1 - int(bw * margin_x))
            new_y1 = max(0, y1 - int(bh * margin_y * 0.3))
            new_x2 = min(w, x2 + int(bw * margin_x))
            new_y2 = min(h, y2 + int(bh * margin_y))

            margin_bboxes.append([new_x1, new_y1, new_x2, new_y2])

        label = BG_ID
        img_id_info = "-1"
        
        u_x1, u_y1, u_x2, u_y2 = None, None, None, None

        if len(margin_bboxes) >= 2:
            u_x1 = min(margin_bboxes[0][0], margin_bboxes[1][0])
            u_y1 = min(margin_bboxes[0][1], margin_bboxes[1][1])
            u_x2 = max(margin_bboxes[0][2], margin_bboxes[1][2])
            u_y2 = max(margin_bboxes[0][3], margin_bboxes[1][3])

        if u_x1 is not None:
            cropped_img_ori = frame_x[max(0, u_y1):min(h, u_y2), max(0, u_x1):min(w, u_x2)]
            cropped_img_gaze = frame_gaze[max(0, u_y1):min(h, u_y2), max(0, u_x1):min(w, u_x2)]
            
            gaze_xy = find_gaze_point(cropped_img_gaze)
            
            if gaze_xy is not None:
                seg_mask = run_segformer(processor, model_segformer, cropped_img_ori)
                x_gaze = np.clip(int(gaze_xy[0]), 0, seg_mask.shape[1] - 1)
                y_gaze = np.clip(int(gaze_xy[1]), 0, seg_mask.shape[0] - 1)
                
                seg_label = seg_mask[y_gaze, x_gaze]
                
                if seg_label == 1 or seg_label == 2:
                    label = int(seg_label)
                    w_cropped = cropped_img_ori.shape[1]
                    cropped_img_arcface = cropped_img_ori[:, :w_cropped // 2]
                    matched_id = run_arcface(cropped_img_arcface, ref_embs)
                    
                    if matched_id is not None:
                        img_id_info = matched_id
                    else:
                        img_id_info = "-1"
                else: 
                    label = BG_ID
                    img_id_info = "-1"
                    
            else: 
                label = BG_ID
                img_id_info = "-1"

        else:
            label = BG_ID
            img_id_info = "-1"

        final_results[f"frame_{frame_idx}"] = {"label": int(label), "img_id": img_id_info}


        if frame_idx in sample_idx and u_x1 is not None:
            text = f"{label}_{img_id_info}"
            org = (10, 30)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (0, 255, 0)
            thickness = 2
            
            sample_img_for_viz = cropped_img_gaze.copy()
            
            if gaze_xy is not None:
                cv2.circle(sample_img_for_viz, (int(gaze_xy[0]), int(gaze_xy[1])), 5, (0, 0, 255), -1)
            
            cv2.putText(sample_img_for_viz, text, org, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
            cv2.imwrite(os.path.join(sample_output_dir, f"frame_{p_num}_{frame_idx}.jpg"), sample_img_for_viz)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap_x.release()
    print(" -> Done processing cropped_ori and gaze", flush=True)


    # with open(f"result_json/exp2_p{p_num}.json", 'w') as f:
    #     json.dump(final_results, f, indent=4)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"exp2_p{p_num}.json")
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=4)

    # break

