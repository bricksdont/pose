#!/usr/bin/env python3
"""
Adapted from https://github.com/T-S-Liang/SDPose-OOD/blob/main/gradio_app/SDPose_gradio.py
"""
import sys
import subprocess
import json
import torch
import os
import numpy as np
import sys
import cv2
import tempfile
import math
import matplotlib
from pathlib import Path

SDPOSEOOD_ROOT = (
    Path(__file__).resolve()
    .parents[1]          # pose_format
    / "third_party"
    / "sdposeood"
)

if not SDPOSEOOD_ROOT.exists():
    print("⚠️  SDPose-OOD not found.")
    print("📥 Cloning SDPose-OOD into:", SDPOSEOOD_ROOT)

    SDPOSEOOD_ROOT.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.check_call([
            "git",
            "clone",
            "https://github.com/T-S-Liang/SDPose-OOD.git",
            str(SDPOSEOOD_ROOT),
        ])
    except FileNotFoundError:
        raise RuntimeError(
            "git is not installed or not available in PATH. "
            "Please install git or manually clone "
            "https://github.com/T-S-Liang/SDPose-OOD "
            f"into {SDPOSEOOD_ROOT}"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Failed to clone SDPose-OOD repository"
        ) from e

# Make SDPose-OOD importable
sys.path.insert(0, str(SDPOSEOOD_ROOT))

from PIL import Image
from torchvision import transforms
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from transformers import CLIPTextModel, CLIPTokenizer
from models.HeatmapHead import get_heatmap_head
from models.ModifiedUNet import Modified_forward
from safetensors.torch import load_file
from pipelines.SDPose_D_Pipeline import SDPose_D_Pipeline

try:
    from diffusers.utils import is_xformers_available
except ImportError:
    def is_xformers_available():
        return False

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not available, YOLO detection will be disabled")

from pose_format.utils.cocowholebody133_header import cocowholebody_components
from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderDimensions


MODEL_PATH="teemosliang/SDPose-Wholebody"  # path to HF repo

def sdpose_components(): 
    return cocowholebody_components()

def estimate_and_load_sdpose(frames: list, 
                fps: float = 24, 
                use_cpu: bool = False, 
                width=1000, 
                height=1000):

    """
    Loads SDPose pose data

    Parameters
    ----------
    frames: list of video frames

    Returns
    -------
    Pose
        Loaded pose data with header and body 
    """
    print("Loading pose with SDPose...")

    dimensions = PoseHeaderDimensions(width=width, height=height)

    header: PoseHeader = PoseHeader(version=0.1,
                                    dimensions=dimensions,
                                    components=sdpose_components())
    
    body: NumPyPoseBody = process_sdpose(frames, fps, use_cpu)

    return Pose(header, body)

def process_sdpose(frames, fps, use_cpu=False):

    inference_engine = SDPoseInference()
    inference_engine.load_model(MODEL_PATH)

    frames_data = []
    frames_conf = []

    for frame_idx, frame in enumerate(frames):

        # do inference
        result_image, all_keypoints, all_scores, info_text, json_file_path = inference_engine.predict_image(
            frame, enable_yolo=True, 
            restore_coords=True, flip_test=False, 
            process_all_persons=False
        )

        if all_keypoints and len(all_keypoints) > 0:
            kpts = all_keypoints[0]  # (K, 2)
            scores = all_scores[0]   # (K,)
            
            # Ensure correct shape
            kpts = np.asarray(kpts, dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32)
            
            frames_data.append(kpts)
            frames_conf.append(scores)
        else:
            # No person detected - add zero keypoints
            num_keypoints = 133  # for wholebody
            frames_data.append(np.zeros((num_keypoints, 2), dtype=np.float32))
            frames_conf.append(np.zeros(num_keypoints, dtype=np.float32))
    
    # DEBUG: Check accumulated data
    print(f"\nTotal frames collected: {len(frames_data)}")
    for i, (kpts, conf) in enumerate(zip(frames_data[:3], frames_conf[:3])):  # Check first 3
        print(f"Frame {i} data shape: {kpts.shape}, conf shape: {conf.shape}")
    
    data = np.asarray(frames_data, dtype=np.float32)
    confidence = np.asarray(frames_conf, dtype=np.float32)
    
    print(f"Final data shape before expand_dims: {data.shape}")
    print(f"Final confidence shape before expand_dims: {confidence.shape}")
    
    # Add people dimension: (frames, K, 2) -> (frames, 1, K, 2)
    data = np.expand_dims(data, axis=1)
    confidence = np.expand_dims(confidence, axis=1)

    return NumPyPoseBody(
        fps=fps,
        data=data,
        confidence=confidence,
    )

'''
def process_sdpose(frames, fps, use_cpu=False):

    inference_engine = SDPoseInference()
    inference_engine.load_model(MODEL_PATH)

    frames_data = []
    frames_conf = []

    for frame in frames:

        # do inference
        result_image, all_keypoints, all_scores, info_text, json_file_path = inference_engine.predict_image(
            frame, enable_yolo=True, 
            restore_coords=True, flip_test=False, 
            process_all_persons=False
        )

        if all_keypoints and len(all_keypoints) > 0:      
            frames_data.append(all_keypoints[0][None, :, :])  # (1, K, 2)
            frames_conf.append(all_scores[0][None, :])        # (1, K)
        else:
            # No person detected - add zero keypoints
            num_keypoints = 133
            frames_data.append(np.zeros((1, num_keypoints, 2), dtype=np.float32))
            frames_conf.append(np.zeros((1, num_keypoints), dtype=np.float32))
        
        print("Keypoints size: ", all_keypoints.shape)
        print("Scores size: ", all_scores.shape)
        print("Keypoints: ", all_keypoints)
        print("All scores:", all_scores)
    
    data = np.asarray(frames_data, dtype=np.float32)
    confidence = np.asarray(frames_conf, dtype=np.float32)

    return NumPyPoseBody(
        fps=fps,
        data=data,
        confidence=confidence,
    )
'''

def detect_person_yolo(image, yolo_model_path="yolov8n.pt", confidence_threshold=0.5):
    """
    Detect person using YOLO
    Returns: List of bboxes [x1, y1, x2, y2] and whether YOLO was used
    """
    if not YOLO_AVAILABLE:
        print("⚠️  YOLO not available, using full image")
        h, w = image.shape[:2]
        return [[0, 0, w, h]], False
    
    try:
        print("🔍 Using YOLO for person detection...")
        
        # Load YOLO model
        if yolo_model_path and os.path.exists(yolo_model_path):
            print(f"   Loading custom YOLO model: {yolo_model_path}")
            model = YOLO(yolo_model_path)
        else:
            print(f"   Loading default YOLOv8n model")
            # Use default YOLOv8
            model = YOLO('yolov8n.pt')
        
        # Run detection
        print(f"   Running YOLO detection on image shape: {image.shape}")
        results = model(image, verbose=False)
        print(f"   YOLO returned {len(results)} result(s)")
        
        # Extract person detections (class 0 is person in COCO)
        person_bboxes = []
        for result in results:
            boxes = result.boxes
            print(f"   Result has {len(boxes) if boxes is not None else 0} boxes")
            if boxes is not None:
                for box in boxes:
                    # Check if it's a person (class 0) and confidence is high enough
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    print(f"   Box: class={cls}, conf={conf:.3f}")
                    if cls == 0 and conf > confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        print(f"   ✓ Person detected: bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                        person_bboxes.append([float(x1), float(y1), float(x2), float(y2), conf])
        
        if person_bboxes:
            # Sort by confidence and return all
            person_bboxes.sort(key=lambda x: x[4], reverse=True)
            bboxes = [bbox[:4] for bbox in person_bboxes]
            print(f"✅ Detected {len(bboxes)} person(s)")
            return bboxes, True
        else:
            print("⚠️  No person detected, using full image")
            h, w = image.shape[:2]
            return [[0, 0, w, h]], False
        
    except Exception as e:
        print(f"⚠️  YOLO detection failed: {e}, using full image")
        h, w = image.shape[:2]
        return [[0, 0, w, h]], False
    
def restore_keypoints_to_original(keypoints, crop_info, input_size, original_size):
    """
    Restore keypoints from cropped/resized space back to original image space
    keypoints: (N, 2) in pixel coordinates of the resized/cropped image
    crop_info: (x1, y1, crop_width, crop_height)
    input_size: (W, H) of the model input
    original_size: (W, H) of the original image
    """
    x1, y1, crop_w, crop_h = crop_info
    input_w, input_h = input_size
    
    # Scale from model input size to crop size
    scale_x = crop_w / input_w
    scale_y = crop_h / input_h
    
    keypoints_restored = keypoints.copy()
    keypoints_restored[:, 0] = keypoints[:, 0] * scale_x + x1
    keypoints_restored[:, 1] = keypoints[:, 1] * scale_y + y1
    
    return keypoints_restored

def preprocess_image_for_sdpose(image, bbox=None, input_size=(768, 1024)):
    """
    Preprocess image for SDPose inference
    Returns: (input_tensor, original_size, crop_info)
    crop_info: (x1, y1, crop_width, crop_height) for coordinate restoration
    """
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR from OpenCV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb)
        original_size = (image.shape[1], image.shape[0])  # (W, H)
    else:
        pil_image = image
        original_size = pil_image.size  # (W, H)
    
    # If bbox is provided, crop the image
    crop_info = None
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        print(f"   📦 Cropping to bbox: [{x1}, {y1}, {x2}, {y2}]")
        # Ensure bbox is within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(pil_image.width, x2)
        y2 = min(pil_image.height, y2)
        
        if x2 > x1 and y2 > y1:
            cropped_image = pil_image.crop((x1, y1, x2, y2))
            crop_info = (x1, y1, x2 - x1, y2 - y1)
            pil_image = cropped_image
            print(f"   ✂️  Cropped image size: {cropped_image.size}")
        else:
            print("⚠️  Invalid bbox, using full image")
            crop_info = (0, 0, pil_image.width, pil_image.height)
    else:
        print(f"   📦 No bbox provided, using full image size: {pil_image.size}")
        crop_info = (0, 0, pil_image.width, pil_image.height)
    
    # Resize to target size
    resized = pil_image.resize(input_size, Image.BILINEAR)
    
    # Apply transforms
    transform_list = [
        transforms.Resize((input_size[1], input_size[0])),  # (H, W)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    val_transform = transforms.Compose(transform_list)
    input_tensor = val_transform(pil_image).unsqueeze(0)
    
    return input_tensor, original_size, crop_info

def draw_wholebody_keypoints_openpose_style(canvas, keypoints, scores=None, threshold=0.3, overlay_mode=False, overlay_alpha=0.6):
    """
    Draw wholebody keypoints (134 keypoints after processing) in DWPose style
    Expected keypoint format (after neck insertion and remapping):
    - Body: 0-17 (18 keypoints in OpenPose format, neck at index 1)
    - Foot: 18-23 (6 keypoints)
    - Face: 24-91 (68 landmarks)
    - Right hand: 92-112 (21 keypoints)
    - Left hand: 113-133 (21 keypoints)
    canvas: The canvas to draw on (should be a black canvas for multi-person, or original image copy for single person)
    overlay_mode: Not used anymore, kept for compatibility
    overlay_alpha: Not used in this function, blending happens outside
    
    Reference: DWPose util.py drawing style
    """
    H, W, C = canvas.shape
    
    # Fixed sizes matching DWPose style
    stickwidth = 4
    
    # Body connections - matching DWPose limbSeq (1-indexed, converted to 0-indexed)
    # Remove shoulder-ear connections: [3,17] and [6,18]
    body_limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18]
    ]
    
    # Hand connections (same for both hands)
    hand_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4],      # thumb
        [0, 5], [5, 6], [6, 7], [7, 8],      # index
        [0, 9], [9, 10], [10, 11], [11, 12], # middle
        [0, 13], [13, 14], [14, 15], [15, 16], # ring
        [0, 17], [17, 18], [18, 19], [19, 20], # pinky
    ]
    
    # Colors matching DWPose
    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
    ]
    
    # Draw body limbs directly on pose_canvas (full opacity)
    if len(keypoints) >= 18:
        for i, limb in enumerate(body_limbSeq):
            # Convert from 1-indexed to 0-indexed
            idx1, idx2 = limb[0] - 1, limb[1] - 1
            
            if idx1 >= 18 or idx2 >= 18:
                continue
            
            if scores is not None:
                if scores[idx1] < threshold or scores[idx2] < threshold:
                    continue
            
            Y = np.array([keypoints[idx1][0], keypoints[idx2][0]])
            X = np.array([keypoints[idx1][1], keypoints[idx2][1]])
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            
            if length < 1:
                continue
            
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])
    
    # Draw body keypoints
    if len(keypoints) >= 18:
        for i in range(18):
            if scores is not None and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 4, colors[i % len(colors)], thickness=-1)
    
    # Draw foot keypoints (18-23, 6 keypoints)
    if len(keypoints) >= 24:
        for i in range(18, 24):
            if scores is not None and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 4, colors[i % len(colors)], thickness=-1)
    
    # Draw right hand (92-112) - DWPose style with cv2.line and HSV colors
    if len(keypoints) >= 113:
        eps = 0.01
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = 92 + edge[0], 92 + edge[1]
            if scores is not None:
                if scores[idx1] < threshold or scores[idx2] < threshold:
                    continue
            
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
            x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                    # HSV to RGB conversion for rainbow colors
                    color = matplotlib.colors.hsv_to_rgb([ie / float(len(hand_edges)), 1.0, 1.0]) * 255
                    cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
        
        # Draw right hand keypoints
        for i in range(92, 113):
            if scores is not None and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    
    # Draw left hand (113-133) - DWPose style with cv2.line and HSV colors
    if len(keypoints) >= 134:
        eps = 0.01
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = 113 + edge[0], 113 + edge[1]
            if scores is not None:
                if scores[idx1] < threshold or scores[idx2] < threshold:
                    continue
            
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
            x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                    # HSV to RGB conversion for rainbow colors
                    color = matplotlib.colors.hsv_to_rgb([ie / float(len(hand_edges)), 1.0, 1.0]) * 255
                    cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
        
        # Draw left hand keypoints
        for i in range(113, 134):
            if scores is not None and i < len(scores) and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    
    # Draw face keypoints (24-91) - DWPose style, white dots only, no lines
    if len(keypoints) >= 92:
        eps = 0.01
        for i in range(24, 92):
            if scores is not None and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    
    return canvas


def draw_body17_keypoints_openpose_style(canvas, keypoints, scores=None, threshold=0.3, overlay_mode=False, overlay_alpha=0.6):
    """
    Draw body keypoints in DWPose style (from util.py draw_bodypose)
    This function converts COCO17 format to OpenPose 18-point format with neck
    Keypoints are in pixel coordinates
    canvas: The canvas to draw on (should be a black canvas for multi-person, or original image copy for single person)
    overlay_mode: Not used anymore, kept for compatibility
    overlay_alpha: Not used in this function, blending happens outside
    """
    H, W, C = canvas.shape
    
    # Compute neck as average of shoulders (index 5 and 6)
    if len(keypoints) >= 7:
        neck = (keypoints[5] + keypoints[6]) / 2
        neck_score = min(scores[5], scores[6]) if scores is not None else 1.0
        
        # Create 18-point format: [nose, neck, rshoulder, relbow, rwrist, lshoulder, lelbow, lwrist, 
        #                          rhip, rknee, rankle, lhip, lknee, lankle, reye, leye, rear, lear]
        # COCO17 indices: [0, -, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        candidate = np.zeros((18, 2))
        candidate_scores = np.zeros(18)
        
        # Map COCO17 to OpenPose 18
        mapping = {
            0: 0,   # nose -> nose
            1: 1,   # neck (computed)
            2: 6,   # right_shoulder -> rshoulder
            3: 8,   # right_elbow -> relbow  
            4: 10,  # right_wrist -> rwrist
            5: 5,   # left_shoulder -> lshoulder
            6: 7,   # left_elbow -> lelbow
            7: 9,   # left_wrist -> lwrist
            8: 12,  # right_hip -> rhip
            9: 14,  # right_knee -> rknee
            10: 16, # right_ankle -> rankle
            11: 11, # left_hip -> lhip
            12: 13, # left_knee -> lknee
            13: 15, # left_ankle -> lankle
            14: 2,  # right_eye -> reye
            15: 1,  # left_eye -> leye
            16: 4,  # right_ear -> rear
            17: 3,  # left_ear -> lear
        }
        
        candidate[0] = keypoints[0]  # nose
        candidate[1] = neck  # neck
        candidate[2] = keypoints[6]  # right_shoulder
        candidate[3] = keypoints[8]  # right_elbow
        candidate[4] = keypoints[10] # right_wrist
        candidate[5] = keypoints[5]  # left_shoulder
        candidate[6] = keypoints[7]  # left_elbow
        candidate[7] = keypoints[9]  # left_wrist
        candidate[8] = keypoints[12] # right_hip
        candidate[9] = keypoints[14] # right_knee
        candidate[10] = keypoints[16]# right_ankle
        candidate[11] = keypoints[11]# left_hip
        candidate[12] = keypoints[13]# left_knee
        candidate[13] = keypoints[15]# left_ankle
        candidate[14] = keypoints[2] # right_eye
        candidate[15] = keypoints[1] # left_eye
        candidate[16] = keypoints[4] # right_ear
        candidate[17] = keypoints[3] # left_ear
        
        if scores is not None:
            candidate_scores[0] = scores[0]
            candidate_scores[1] = neck_score
            candidate_scores[2] = scores[6]
            candidate_scores[3] = scores[8]
            candidate_scores[4] = scores[10]
            candidate_scores[5] = scores[5]
            candidate_scores[6] = scores[7]
            candidate_scores[7] = scores[9]
            candidate_scores[8] = scores[12]
            candidate_scores[9] = scores[14]
            candidate_scores[10] = scores[16]
            candidate_scores[11] = scores[11]
            candidate_scores[12] = scores[13]
            candidate_scores[13] = scores[15]
            candidate_scores[14] = scores[2]
            candidate_scores[15] = scores[1]
            candidate_scores[16] = scores[4]
            candidate_scores[17] = scores[3]
    else:
        return canvas
    
    # Scale stickwidth and circle size based on image resolution
    # Use average of height and width as reference
    avg_size = (H + W) / 2
    stickwidth = max(1, int(avg_size / 256))  # Base reference: 256px -> width 4
    circle_radius = max(2, int(avg_size / 192))  # Base reference: 256px -> radius 5
    
    # DWPose limbSeq (1-indexed, so we subtract 1)
    # Removed [3, 17] and [6, 18] (shoulder-ear connections are redundant)
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18]
    ]
    
    # Colors from DWPose util.py draw_bodypose
    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]
    
    # Draw limbs directly on pose_canvas (full opacity)
    for i in range(len(limbSeq)):
        index = np.array(limbSeq[i]) - 1  # Convert to 0-indexed
        if index[0] >= len(candidate) or index[1] >= len(candidate):
            continue
            
        if scores is not None:
            if candidate_scores[index[0]] < threshold or candidate_scores[index[1]] < threshold:
                continue
        
        Y = candidate[index.astype(int), 0]  # x coordinates
        X = candidate[index.astype(int), 1]  # y coordinates
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        
        if length < 1:
            continue
            
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])
    
    # Draw keypoints
    for i in range(18):
        if scores is not None and candidate_scores[i] < threshold:
            continue
            
        x, y = candidate[i]
        x = int(x)
        y = int(y)
        
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
            
        cv2.circle(canvas, (int(x), int(y)), circle_radius, colors[i % len(colors)], thickness=-1)
    
    return canvas

def convert_to_openpose_json(all_keypoints, all_scores, image_width, image_height, keypoint_scheme="wholebody"):
    """
    Convert keypoints to OpenPose JSON format
    Args:
        all_keypoints: List of keypoints for each person, shape (N_people, K, 2)
        all_scores: List of scores for each person, shape (N_people, K)
        image_width: Original image width
        image_height: Original image height
        keypoint_scheme: "body" or "wholebody"
    Returns:
        Dictionary in OpenPose JSON format
    """
    people = []
    
    for person_idx, (keypoints, scores) in enumerate(zip(all_keypoints, all_scores)):
        person_data = {}
        
        if keypoint_scheme == "body":
            # Body only: 17 keypoints
            pose_kpts = []
            for i in range(min(17, len(keypoints))):
                pose_kpts.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])])
            
            # Pad if needed
            while len(pose_kpts) < 17 * 3:
                pose_kpts.extend([0.0, 0.0, 0.0])
            
            person_data["pose_keypoints_2d"] = pose_kpts
            person_data["hand_left_keypoints_2d"] = [0.0] * 63
            person_data["hand_right_keypoints_2d"] = [0.0] * 63
            person_data["face_keypoints_2d"] = [0.0] * 204
            person_data["foot_keypoints_2d"] = [0.0] * 18
            
        else:
            # Wholebody: 133 keypoints (after processing)
            # Body: 0-17 (18 keypoints including neck at index 17)
            # Foot: 18-23 (6 keypoints)
            # Face: 24-91 (68 keypoints)
            # Right hand: 92-112 (21 keypoints)
            # Left hand: 113-133 (21 keypoints)
            
            # Body keypoints (18 including neck)
            pose_kpts = []
            for i in range(min(18, len(keypoints))):
                pose_kpts.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])])
            while len(pose_kpts) < 18 * 3:
                pose_kpts.extend([0.0, 0.0, 0.0])
            person_data["pose_keypoints_2d"] = pose_kpts
            
            # Foot keypoints (6)
            foot_kpts = []
            for i in range(18, min(24, len(keypoints))):
                foot_kpts.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])])
            while len(foot_kpts) < 6 * 3:
                foot_kpts.extend([0.0, 0.0, 0.0])
            person_data["foot_keypoints_2d"] = foot_kpts
            
            # Face keypoints (68)
            face_kpts = []
            for i in range(24, min(92, len(keypoints))):
                face_kpts.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])])
            while len(face_kpts) < 68 * 3:
                face_kpts.extend([0.0, 0.0, 0.0])
            person_data["face_keypoints_2d"] = face_kpts
            
            # Right hand keypoints (21)
            right_hand_kpts = []
            for i in range(92, min(113, len(keypoints))):
                right_hand_kpts.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])])
            while len(right_hand_kpts) < 21 * 3:
                right_hand_kpts.extend([0.0, 0.0, 0.0])
            person_data["hand_right_keypoints_2d"] = right_hand_kpts
            
            # Left hand keypoints (21)
            left_hand_kpts = []
            for i in range(113, min(134, len(keypoints))):
                left_hand_kpts.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])])
            while len(left_hand_kpts) < 21 * 3:
                left_hand_kpts.extend([0.0, 0.0, 0.0])
            person_data["hand_left_keypoints_2d"] = left_hand_kpts
        
        people.append(person_data)
    
    result = {
        "people": people,
        "canvas_width": int(image_width),
        "canvas_height": int(image_height)
    }
    
    return result



class SDPoseInference:
    """SDPose inference class supporting both body and wholebody schemes"""
    
    def __init__(self):
        self.pipeline = None
        self.device = None
        self.model_loaded = False
        self.keypoint_scheme = "wholebody"  # "body" or "wholebody"
        self.input_size = (768, 1024)  # (W, H)
        
    def load_model(self, model_path, keypoint_scheme="wholebody", device="auto"):
        """Load the SDPose model"""
        try:
            if device == "auto":
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            
            self.keypoint_scheme = keypoint_scheme
            
            print(f"Loading model on device: {self.device}")
            print(f"Keypoint scheme: {keypoint_scheme}")
            print(f"Loading from: {model_path}")
            
            # Load UNet
            unet = UNet2DConditionModel.from_pretrained(
                model_path, subfolder="unet", revision=None,
                class_embed_type="projection", projection_class_embeddings_input_dim=4,
                low_cpu_mem_usage=False, device_map=None,
            )
            
            # Apply modifications
            unet = Modified_forward(unet, keypoint_scheme=keypoint_scheme)
            
            # Load other components
            vae = AutoencoderKL.from_pretrained(model_path, subfolder='vae')
            tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer')
            text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder='text_encoder')
            
            # Load decoder
            print("Downloading decoder weights from Hugging Face...")
            hm_decoder = get_heatmap_head(mode=keypoint_scheme)
            dec_path = hf_hub_download(
                repo_id=model_path,
                filename="decoder/decoder.safetensors",
                repo_type="model"
            )
            print(f"✓ Decoder downloaded to: {dec_path}")
            hm_decoder.load_state_dict(
                load_file(dec_path, device="cpu"),
                strict=True
            )
            print("✓ Decoder weights loaded")

            # Load scheduler
            noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder='scheduler')
            
            # Move to device
            unet = unet.to(self.device)
            vae = vae.to(self.device)
            text_encoder = text_encoder.to(self.device)
            hm_decoder = hm_decoder.to(self.device)
            
            # Create pipeline
            self.pipeline = SDPose_D_Pipeline(
                unet=unet,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                scheduler=noise_scheduler,
                decoder=hm_decoder
            )
            
            # Enable xformers if available
            if is_xformers_available():
                try:
                    self.pipeline.unet.enable_xformers_memory_efficient_attention()
                    print("✓ xformers enabled")
                except Exception as e:
                    print(f"⚠️  Could not enable xformers: {e}")
            
            self.model_loaded = True
            print("✓ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
      
    def predict_image(self, image, enable_yolo=True, yolo_model_path=None, 
                     score_threshold=0.3, restore_coords=True, flip_test=False, process_all_persons=True, overlay_alpha=0.6):
        """
        Run inference on a single image (supports multi-person)
        overlay_alpha: Opacity of pose+black background layer (0.0=invisible, 1.0=fully opaque)
        Returns: (result_image, keypoints, scores, info_text, json_file_path)
        """
        if not self.model_loaded or self.pipeline is None:
            return None, None, None, "Model not loaded. Please load the model first.", None
        
        try:
            # Handle image format: Gradio Image(type="numpy") returns RGB numpy array
            if isinstance(image, np.ndarray):
                original_image_rgb = image.copy()
            else:
                original_image_rgb = np.array(image)
            
            # Convert to BGR for YOLO (YOLO expects BGR)
            original_image_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
            
            # Step 1: Person detection (if enabled)
            bboxes_list = []
            detection_info = ""
            if enable_yolo:
                print(f"🔍 YOLO detection enabled (yolo_model_path: {yolo_model_path})")
                bboxes, used_yolo = detect_person_yolo(original_image_bgr, yolo_model_path, confidence_threshold=0.5)
                print(f"   YOLO actually used: {used_yolo}, detected {len(bboxes)} person(s)")
                if bboxes and len(bboxes) > 0:
                    bboxes_list = bboxes if process_all_persons else [bboxes[0]]
                    detection_info = f"Detected {len(bboxes)} person(s) by YOLO, processing {len(bboxes_list)}"
                    print(f"✅ {detection_info}")
                else:
                    bboxes_list = [None]  # Process full image
                    detection_info = "No person detected by YOLO, using full image"
                    print(f"⚠️  {detection_info}")
            else:
                bboxes_list = [None]  # Process full image
                detection_info = "YOLO disabled, using full image"
                print(f"⚠️  {detection_info}")
            
            # Step 2-6: Process each person
            # Create black canvas for all pose drawings
            pose_canvas = np.zeros_like(original_image_rgb)
            all_keypoints = []
            all_scores = []
            
            for person_idx, bbox in enumerate(bboxes_list):
                print(f"\n👤 Processing person {person_idx + 1}/{len(bboxes_list)}")
                
                # Step 2: Preprocess image
                print("🔄 Preprocessing image...")
                print(f"   📦 Bbox: {bbox}")
                input_tensor, original_size, crop_info = preprocess_image_for_sdpose(
                    original_image_bgr, bbox, self.input_size
                )
                print(f"   ✂️  Crop info: {crop_info}")
                input_tensor = input_tensor.to(self.device)

                print("Model input tensor shape:", input_tensor.shape)
                
                # Step 3: Run inference
                print("🚀 Running SDPose inference...")
                test_cfg = {'flip_test': False}
                
                with torch.no_grad():
                    out = self.pipeline(
                        input_tensor,
                        timesteps=[999],
                        test_cfg=test_cfg,
                        show_progress_bar=False,
                        mode="inference",
                    )
                    
                    # Extract keypoints and scores
                    heatmap_inst = out[0]
                    keypoints = heatmap_inst.keypoints[0]  # (K, 2)
                    scores = heatmap_inst.keypoint_scores[0]  # (K,)
                    
                    # Convert to numpy
                    if torch.is_tensor(keypoints):
                        keypoints = keypoints.cpu().numpy()
                    if torch.is_tensor(scores):
                        scores = scores.cpu().numpy()
                
                print(f"📊 Detected {len(keypoints)} keypoints")
                
                # Step 4: Restore coordinates to original space
                if restore_coords and bbox is not None:
                    keypoints_original = restore_keypoints_to_original(
                        keypoints, crop_info, self.input_size, original_size
                    )
                else:
                    scale_x = original_size[0] / self.input_size[0]
                    scale_y = original_size[1] / self.input_size[1]
                    keypoints_original = keypoints.copy()
                    keypoints_original[:, 0] *= scale_x
                    keypoints_original[:, 1] *= scale_y
                
                all_keypoints.append(keypoints_original)
                all_scores.append(scores)
                
                # Step 5: Draw keypoints for this person
                print(f"🎨 Drawing keypoints for person {person_idx + 1}...")
                
                if self.keypoint_scheme == "body":
                    if len(keypoints_original) >= 17:
                        # Draw on pose_canvas (black background, shared by all persons)
                        pose_canvas = draw_body17_keypoints_openpose_style(
                            pose_canvas, keypoints_original[:17], scores[:17], 
                            threshold=score_threshold
                        )
                else:
                    # Wholebody scheme
                    keypoints_with_neck = keypoints_original.copy()
                    scores_with_neck = scores.copy()
                    
                    if len(keypoints_original) >= 17:
                        neck = (keypoints_original[5] + keypoints_original[6]) / 2
                        neck_score = min(scores[5], scores[6]) if scores[5] > 0.3 and scores[6] > 0.3 else 0
                        
                        keypoints_with_neck = np.insert(keypoints_original, 17, neck, axis=0)
                        scores_with_neck = np.insert(scores, 17, neck_score)
                        
                        mmpose_idx = np.array([17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3])
                        openpose_idx = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17])
                        
                        temp_kpts = keypoints_with_neck.copy()
                        temp_scores = scores_with_neck.copy()
                        temp_kpts[openpose_idx] = keypoints_with_neck[mmpose_idx]
                        temp_scores[openpose_idx] = scores_with_neck[mmpose_idx]
                        
                        keypoints_with_neck = temp_kpts
                        scores_with_neck = temp_scores
                    
                    # Draw on pose_canvas (black background, shared by all persons)
                    pose_canvas = draw_wholebody_keypoints_openpose_style(
                        pose_canvas, keypoints_with_neck, scores_with_neck, 
                        threshold=score_threshold
                    )
            
            # Blend original image with pose canvas after all persons are drawn
            # overlay_alpha: transparency of (pose + black background) layer
            # 0.0 = invisible (only original image), 1.0 = fully opaque (pose + black bg)
            result_image = cv2.addWeighted(original_image_rgb, 1.0 - overlay_alpha, pose_canvas, overlay_alpha, 0)
            
            # Create info text
            info_text = self._create_info_text(
                original_size, self.input_size, detection_info, bboxes_list[0] if len(bboxes_list) == 1 else None,
                all_keypoints[0] if len(all_keypoints) > 0 else None, 
                all_scores[0] if len(all_scores) > 0 else None, 
                score_threshold,
                len(bboxes_list)
            )
            
            # Generate JSON file
            json_file_path = None
            if all_keypoints and len(all_keypoints) > 0:
                try:
                    # Convert to OpenPose JSON format
                    json_data = convert_to_openpose_json(
                        all_keypoints, all_scores, 
                        original_size[0], original_size[1],
                        self.keypoint_scheme
                    )
                    
                    # Save to temporary file
                    temp_json = tempfile.NamedTemporaryFile(
                        mode='w', suffix='.json', delete=False, 
                        dir=tempfile.gettempdir()
                    )
                    json.dump(json_data, temp_json, indent=2)
                    json_file_path = temp_json.name
                    temp_json.close()
                    
                    print(f"✅ JSON file saved: {json_file_path}")
                    
                except Exception as e:
                    print(f"⚠️  Failed to generate JSON file: {e}")
                    json_file_path = None
            
            print(f"✅ Inference complete. Returning RGB result_image with shape: {result_image.shape}")
            return result_image, all_keypoints, all_scores, info_text, json_file_path
                
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return image, None, None, f"Error during inference: {str(e)}", None
    
    def predict_video(self, video_path, output_path, enable_yolo=True, 
                     yolo_model_path=None, score_threshold=0.3, flip_test=False, overlay_alpha=0.6):
        """
        Run inference on a video file
        overlay_alpha: Opacity of pose+black background layer (0.0=invisible, 1.0=fully opaque)
        Returns: (output_video_path, info_text)
        """
        if not self.model_loaded or self.pipeline is None:
            return None, "Model not loaded. Please load the model first."
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, f"Error: Could not open video {video_path}"
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:
                fps = 30  # Default fallback
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📹 Processing video: {total_frames} frames at {fps} FPS, size {width}x{height}")
            
            # Create video writer
            # Use mp4v for initial encoding (will re-encode to H.264 later if needed)
            print(f"📝 Creating VideoWriter with mp4v codec...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Ensure output path has .mp4 extension
            actual_output_path = output_path
            if not actual_output_path.endswith('.mp4'):
                actual_output_path = output_path.rsplit('.', 1)[0] + '.mp4'
            
            out = cv2.VideoWriter(actual_output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                cap.release()
                print(f"❌ Failed to open VideoWriter")
                return None, f"Error: Could not create video writer"
            
            print(f"✅ VideoWriter opened successfully: {actual_output_path}")
            
            frame_count = 0
            processed_count = 0
            
            # Process each frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Convert frame from BGR to RGB for predict_image
                # cv2.VideoCapture reads in BGR format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run inference on frame (frame_rgb is RGB)
                # Process all detected persons
                result_frame, _, _, _, _ = self.predict_image(
                    frame_rgb, enable_yolo=enable_yolo, yolo_model_path=yolo_model_path,
                    score_threshold=score_threshold, restore_coords=True, flip_test=flip_test, 
                    process_all_persons=True, overlay_alpha=overlay_alpha
                )
                
                if result_frame is not None:
                    # result_frame is RGB from predict_image, convert to BGR for video writing
                    result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                    
                    # Check frame size matches
                    if result_frame_bgr.shape[:2] != (height, width):
                        print(f"⚠️  Frame size mismatch: {result_frame_bgr.shape[:2]} vs expected ({height}, {width}), resizing...")
                        result_frame_bgr = cv2.resize(result_frame_bgr, (width, height))
                    
                    out.write(result_frame_bgr)
                    processed_count += 1
                else:
                    # If inference failed, write original frame (already BGR)
                    print(f"⚠️  Frame {frame_count} inference failed, using original")
                    out.write(frame)
                
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames, written {processed_count}")
            
            cap.release()
            out.release()
            
            # Ensure the video file is properly written and flushed
            # Small delay to ensure file system has finished writing
            import time
            time.sleep(0.5)
            
            # Verify the output file exists and has content
            if not os.path.exists(actual_output_path):
                return None, f"Error: Output video file was not created at {actual_output_path}"
            
            file_size = os.path.getsize(actual_output_path)
            if file_size == 0:
                return None, f"Error: Output video file is empty (0 bytes)"
            
            print(f"✅ Video file created: {actual_output_path} ({file_size} bytes)")
            
            # If we used mp4v codec, try to re-encode to H.264 for better browser compatibility
            final_output_path = actual_output_path
            if actual_output_path.endswith('.mp4'):
                try:
                    import subprocess
                    print("🔄 Re-encoding video to H.264 for better browser compatibility...")
                    
                    # Create a new temp file for H.264 version
                    h264_path = actual_output_path.rsplit('.', 1)[0] + '_h264.mp4'
                    
                    # Use ffmpeg to re-encode
                    cmd = [
                        'ffmpeg', '-y', '-i', actual_output_path,
                        '-c:v', 'libx264', '-preset', 'fast', 
                        '-crf', '23', '-pix_fmt', 'yuv420p',
                        h264_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, timeout=300)
                    
                    if result.returncode == 0 and os.path.exists(h264_path):
                        h264_size = os.path.getsize(h264_path)
                        if h264_size > 0:
                            print(f"✅ Re-encoded to H.264: {h264_path} ({h264_size} bytes)")
                            # Use the H.264 version
                            final_output_path = h264_path
                            file_size = h264_size
                            # Remove the original mp4v version
                            try:
                                os.unlink(actual_output_path)
                            except:
                                pass
                        else:
                            print(f"⚠️  Re-encoded file is empty, using original")
                    else:
                        print(f"⚠️  Re-encoding failed, using original mp4v version")
                        if result.stderr:
                            print(f"   ffmpeg error: {result.stderr.decode()[:200]}")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Re-encoding timed out, using original")
                except Exception as e:
                    print(f"⚠️  Re-encoding failed: {e}, using original")
            
            info_text = f"✅ Video processing complete!\n"
            info_text += f"📊 Total frames: {total_frames}\n"
            info_text += f"✓ Processed: {processed_count}\n"
            info_text += f"🎞️ FPS: {fps}\n"
            info_text += f"📏 Resolution: {width}x{height}\n"
            info_text += f"💾 File size: {file_size / (1024*1024):.2f} MB\n"
            info_text += f"💾 Output saved to: {final_output_path}"
            
            print(info_text)
            return final_output_path, info_text
            
        except Exception as e:
            print(f"Error during video inference: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error during video inference: {str(e)}"
        
    def _create_info_text(self, original_size, input_size, detection_info, bbox,
                        keypoints, scores, threshold, num_persons=1):
        """Create informative text about the inference results"""
        info_text = "🎯 SDPose Keypoint Detection Results\n" + "="*60 + "\n"
        info_text += f"📏 Original Image Size: {original_size}\n"
        info_text += f"🔧 Model Input Size: {input_size}\n"
        info_text += f"🧠 Keypoint Scheme: {self.keypoint_scheme}\n"
        info_text += f"🔍 Detection: {detection_info}\n"
        info_text += f"👥 Number of Persons Processed: {num_persons}\n"
        if bbox:
            info_text += f"📦 Bounding Box (first person): [{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]\n"
        info_text += f"🎚️ Score Threshold: {threshold}\n"
        info_text += "="*60 + "\n\n"
        
        # Count detected keypoints (for first person if available)
        if keypoints is not None and scores is not None:
            detected_count = np.sum(scores >= threshold)
            total_count = len(scores)
            info_text += f"📊 Summary (first person): {detected_count}/{total_count} keypoints detected above threshold\n"
        
        info_text += f"🎨 Visualization: Openpose style (similar to DWPose)\n"
        info_text += f"📍 Coordinates: Restored to original image space\n"
        
        return info_text