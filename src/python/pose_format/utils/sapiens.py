import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderDimensions, PoseHeaderComponent


# ──────────────────────────────────────────────────────────────────────
# Goliath 308 keypoints (teeth removed) – ordered to match model output
# ──────────────────────────────────────────────────────────────────────

BODY_POINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]

RIGHT_HAND_POINTS = [
    "right_thumb4", "right_thumb3", "right_thumb2", "right_thumb_third_joint",
    "right_forefinger4", "right_forefinger3", "right_forefinger2", "right_forefinger_third_joint",
    "right_middle_finger4", "right_middle_finger3", "right_middle_finger2", "right_middle_finger_third_joint",
    "right_ring_finger4", "right_ring_finger3", "right_ring_finger2", "right_ring_finger_third_joint",
    "right_pinky_finger4", "right_pinky_finger3", "right_pinky_finger2", "right_pinky_finger_third_joint",
    "right_wrist",
]

LEFT_HAND_POINTS = [
    "left_thumb4", "left_thumb3", "left_thumb2", "left_thumb_third_joint",
    "left_forefinger4", "left_forefinger3", "left_forefinger2", "left_forefinger_third_joint",
    "left_middle_finger4", "left_middle_finger3", "left_middle_finger2", "left_middle_finger_third_joint",
    "left_ring_finger4", "left_ring_finger3", "left_ring_finger2", "left_ring_finger_third_joint",
    "left_pinky_finger4", "left_pinky_finger3", "left_pinky_finger2", "left_pinky_finger_third_joint",
    "left_wrist",
]

BODY_EXTRA_POINTS = [
    "left_olecranon", "right_olecranon",
    "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion",
    "neck",
]

FACE_POINTS = [
    # Nose bridge & chin (70-77)
    "center_of_glabella", "center_of_nose_root", "tip_of_nose_bridge",
    "midpoint_1_of_nose_bridge", "midpoint_2_of_nose_bridge", "midpoint_3_of_nose_bridge",
    "center_of_labiomental_groove", "tip_of_chin",
    # Right eyebrow (78-86)
    "upper_startpoint_of_r_eyebrow", "lower_startpoint_of_r_eyebrow", "end_of_r_eyebrow",
    "upper_midpoint_1_of_r_eyebrow", "lower_midpoint_1_of_r_eyebrow",
    "upper_midpoint_2_of_r_eyebrow", "upper_midpoint_3_of_r_eyebrow",
    "lower_midpoint_2_of_r_eyebrow", "lower_midpoint_3_of_r_eyebrow",
    # Left eyebrow (87-95)
    "upper_startpoint_of_l_eyebrow", "lower_startpoint_of_l_eyebrow", "end_of_l_eyebrow",
    "upper_midpoint_1_of_l_eyebrow", "lower_midpoint_1_of_l_eyebrow",
    "upper_midpoint_2_of_l_eyebrow", "upper_midpoint_3_of_l_eyebrow",
    "lower_midpoint_2_of_l_eyebrow", "lower_midpoint_3_of_l_eyebrow",
    # Left upper eyelid / lash / crease (96-119)
    "l_inner_end_of_upper_lash_line", "l_outer_end_of_upper_lash_line",
    "l_centerpoint_of_upper_lash_line", "l_midpoint_2_of_upper_lash_line",
    "l_midpoint_1_of_upper_lash_line", "l_midpoint_6_of_upper_lash_line",
    "l_midpoint_5_of_upper_lash_line", "l_midpoint_4_of_upper_lash_line",
    "l_midpoint_3_of_upper_lash_line",
    "l_outer_end_of_upper_eyelid_line", "l_midpoint_6_of_upper_eyelid_line",
    "l_midpoint_2_of_upper_eyelid_line", "l_midpoint_5_of_upper_eyelid_line",
    "l_centerpoint_of_upper_eyelid_line", "l_midpoint_4_of_upper_eyelid_line",
    "l_midpoint_1_of_upper_eyelid_line", "l_midpoint_3_of_upper_eyelid_line",
    "l_midpoint_6_of_upper_crease_line", "l_midpoint_2_of_upper_crease_line",
    "l_midpoint_5_of_upper_crease_line", "l_centerpoint_of_upper_crease_line",
    "l_midpoint_4_of_upper_crease_line", "l_midpoint_1_of_upper_crease_line",
    "l_midpoint_3_of_upper_crease_line",
    # Right upper eyelid / lash / crease (120-143)
    "r_inner_end_of_upper_lash_line", "r_outer_end_of_upper_lash_line",
    "r_centerpoint_of_upper_lash_line", "r_midpoint_1_of_upper_lash_line",
    "r_midpoint_2_of_upper_lash_line", "r_midpoint_3_of_upper_lash_line",
    "r_midpoint_4_of_upper_lash_line", "r_midpoint_5_of_upper_lash_line",
    "r_midpoint_6_of_upper_lash_line",
    "r_outer_end_of_upper_eyelid_line", "r_midpoint_3_of_upper_eyelid_line",
    "r_midpoint_1_of_upper_eyelid_line", "r_midpoint_4_of_upper_eyelid_line",
    "r_centerpoint_of_upper_eyelid_line", "r_midpoint_5_of_upper_eyelid_line",
    "r_midpoint_2_of_upper_eyelid_line", "r_midpoint_6_of_upper_eyelid_line",
    "r_midpoint_3_of_upper_crease_line", "r_midpoint_1_of_upper_crease_line",
    "r_midpoint_4_of_upper_crease_line", "r_centerpoint_of_upper_crease_line",
    "r_midpoint_5_of_upper_crease_line", "r_midpoint_2_of_upper_crease_line",
    "r_midpoint_6_of_upper_crease_line",
    # Left lower eyelid / lash (144-160)
    "l_inner_end_of_lower_lash_line", "l_outer_end_of_lower_lash_line",
    "l_centerpoint_of_lower_lash_line", "l_midpoint_2_of_lower_lash_line",
    "l_midpoint_1_of_lower_lash_line", "l_midpoint_6_of_lower_lash_line",
    "l_midpoint_5_of_lower_lash_line", "l_midpoint_4_of_lower_lash_line",
    "l_midpoint_3_of_lower_lash_line",
    "l_outer_end_of_lower_eyelid_line", "l_midpoint_6_of_lower_eyelid_line",
    "l_midpoint_2_of_lower_eyelid_line", "l_midpoint_5_of_lower_eyelid_line",
    "l_centerpoint_of_lower_eyelid_line", "l_midpoint_4_of_lower_eyelid_line",
    "l_midpoint_1_of_lower_eyelid_line", "l_midpoint_3_of_lower_eyelid_line",
    # Right lower eyelid / lash (161-177)
    "r_inner_end_of_lower_lash_line", "r_outer_end_of_lower_lash_line",
    "r_centerpoint_of_lower_lash_line", "r_midpoint_1_of_lower_lash_line",
    "r_midpoint_2_of_lower_lash_line", "r_midpoint_3_of_lower_lash_line",
    "r_midpoint_4_of_lower_lash_line", "r_midpoint_5_of_lower_lash_line",
    "r_midpoint_6_of_lower_lash_line",
    "r_outer_end_of_lower_eyelid_line", "r_midpoint_3_of_lower_eyelid_line",
    "r_midpoint_1_of_lower_eyelid_line", "r_midpoint_4_of_lower_eyelid_line",
    "r_centerpoint_of_lower_eyelid_line", "r_midpoint_5_of_lower_eyelid_line",
    "r_midpoint_2_of_lower_eyelid_line", "r_midpoint_6_of_lower_eyelid_line",
    # Nose details (178-187)
    "tip_of_nose", "bottom_center_of_nose",
    "r_outer_corner_of_nose", "l_outer_corner_of_nose",
    "inner_corner_of_r_nostril", "outer_corner_of_r_nostril", "upper_corner_of_r_nostril",
    "inner_corner_of_l_nostril", "outer_corner_of_l_nostril", "upper_corner_of_l_nostril",
    # Outer mouth / lips (188-203)
    "r_outer_corner_of_mouth", "l_outer_corner_of_mouth",
    "center_of_cupid_bow", "center_of_lower_outer_lip",
    "midpoint_1_of_upper_outer_lip", "midpoint_2_of_upper_outer_lip",
    "midpoint_1_of_lower_outer_lip", "midpoint_2_of_lower_outer_lip",
    "midpoint_3_of_upper_outer_lip", "midpoint_4_of_upper_outer_lip",
    "midpoint_5_of_upper_outer_lip", "midpoint_6_of_upper_outer_lip",
    "midpoint_3_of_lower_outer_lip", "midpoint_4_of_lower_outer_lip",
    "midpoint_5_of_lower_outer_lip", "midpoint_6_of_lower_outer_lip",
    # Inner mouth / lips (204-219)
    "r_inner_corner_of_mouth", "l_inner_corner_of_mouth",
    "center_of_upper_inner_lip", "center_of_lower_inner_lip",
    "midpoint_1_of_upper_inner_lip", "midpoint_2_of_upper_inner_lip",
    "midpoint_1_of_lower_inner_lip", "midpoint_2_of_lower_inner_lip",
    "midpoint_3_of_upper_inner_lip", "midpoint_4_of_upper_inner_lip",
    "midpoint_5_of_upper_inner_lip", "midpoint_6_of_upper_inner_lip",
    "midpoint_3_of_lower_inner_lip", "midpoint_4_of_lower_inner_lip",
    "midpoint_5_of_lower_inner_lip", "midpoint_6_of_lower_inner_lip",
]

LEFT_EAR_POINTS = [
    "l_top_end_of_inferior_crus", "l_top_end_of_superior_crus",
    "l_start_of_antihelix", "l_end_of_antihelix",
    "l_midpoint_1_of_antihelix", "l_midpoint_1_of_inferior_crus",
    "l_midpoint_2_of_antihelix", "l_midpoint_3_of_antihelix",
    "l_point_1_of_inner_helix", "l_point_2_of_inner_helix",
    "l_point_3_of_inner_helix", "l_point_4_of_inner_helix",
    "l_point_5_of_inner_helix", "l_point_6_of_inner_helix",
    "l_point_7_of_inner_helix",
    "l_highest_point_of_antitragus",
    "l_bottom_point_of_tragus", "l_protruding_point_of_tragus",
    "l_top_point_of_tragus", "l_start_point_of_crus_of_helix",
    "l_deepest_point_of_concha", "l_tip_of_ear_lobe",
    "l_midpoint_between_22_15",
    "l_bottom_connecting_point_of_ear_lobe",
    "l_top_connecting_point_of_helix",
    "l_point_8_of_inner_helix",
]

RIGHT_EAR_POINTS = [
    "r_top_end_of_inferior_crus", "r_top_end_of_superior_crus",
    "r_start_of_antihelix", "r_end_of_antihelix",
    "r_midpoint_1_of_antihelix", "r_midpoint_1_of_inferior_crus",
    "r_midpoint_2_of_antihelix", "r_midpoint_3_of_antihelix",
    "r_point_1_of_inner_helix", "r_point_8_of_inner_helix",
    "r_point_3_of_inner_helix", "r_point_4_of_inner_helix",
    "r_point_5_of_inner_helix", "r_point_6_of_inner_helix",
    "r_point_7_of_inner_helix",
    "r_highest_point_of_antitragus",
    "r_bottom_point_of_tragus", "r_protruding_point_of_tragus",
    "r_top_point_of_tragus", "r_start_point_of_crus_of_helix",
    "r_deepest_point_of_concha", "r_tip_of_ear_lobe",
    "r_midpoint_between_22_15",
    "r_bottom_connecting_point_of_ear_lobe",
    "r_top_connecting_point_of_helix",
    "r_point_2_of_inner_helix",
]

LEFT_IRIS_POINTS = [
    "l_center_of_iris",
    "l_border_of_iris_3", "l_border_of_iris_midpoint_1",
    "l_border_of_iris_12", "l_border_of_iris_midpoint_4",
    "l_border_of_iris_9", "l_border_of_iris_midpoint_3",
    "l_border_of_iris_6", "l_border_of_iris_midpoint_2",
]

RIGHT_IRIS_POINTS = [
    "r_center_of_iris",
    "r_border_of_iris_3", "r_border_of_iris_midpoint_1",
    "r_border_of_iris_12", "r_border_of_iris_midpoint_4",
    "r_border_of_iris_9", "r_border_of_iris_midpoint_3",
    "r_border_of_iris_6", "r_border_of_iris_midpoint_2",
]

LEFT_PUPIL_POINTS = [
    "l_center_of_pupil",
    "l_border_of_pupil_3", "l_border_of_pupil_midpoint_1",
    "l_border_of_pupil_12", "l_border_of_pupil_midpoint_4",
    "l_border_of_pupil_9", "l_border_of_pupil_midpoint_3",
    "l_border_of_pupil_6", "l_border_of_pupil_midpoint_2",
]

RIGHT_PUPIL_POINTS = [
    "r_center_of_pupil",
    "r_border_of_pupil_3", "r_border_of_pupil_midpoint_1",
    "r_border_of_pupil_12", "r_border_of_pupil_midpoint_4",
    "r_border_of_pupil_9", "r_border_of_pupil_midpoint_3",
    "r_border_of_pupil_6", "r_border_of_pupil_midpoint_2",
]


# ──────────────────────────────────────────────────────────────────────
# Limb definitions (intra-component only)
# ──────────────────────────────────────────────────────────────────────

def _map_limbs(points, limb_names):
    index_map = {name: idx for idx, name in enumerate(points)}
    return [(index_map[a], index_map[b]) for a, b in limb_names]


BODY_LIMBS_NAMES = [
    ("left_ankle", "left_knee"), ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"), ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_eye", "right_eye"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"),
    ("left_ankle", "left_big_toe"), ("left_ankle", "left_small_toe"), ("left_ankle", "left_heel"),
    ("right_ankle", "right_big_toe"), ("right_ankle", "right_small_toe"), ("right_ankle", "right_heel"),
]

_HAND_FINGER_LIMBS = [
    ("_wrist", "_thumb_third_joint"), ("_thumb_third_joint", "_thumb2"),
    ("_thumb2", "_thumb3"), ("_thumb3", "_thumb4"),
    ("_wrist", "_forefinger_third_joint"), ("_forefinger_third_joint", "_forefinger2"),
    ("_forefinger2", "_forefinger3"), ("_forefinger3", "_forefinger4"),
    ("_wrist", "_middle_finger_third_joint"), ("_middle_finger_third_joint", "_middle_finger2"),
    ("_middle_finger2", "_middle_finger3"), ("_middle_finger3", "_middle_finger4"),
    ("_wrist", "_ring_finger_third_joint"), ("_ring_finger_third_joint", "_ring_finger2"),
    ("_ring_finger2", "_ring_finger3"), ("_ring_finger3", "_ring_finger4"),
    ("_wrist", "_pinky_finger_third_joint"), ("_pinky_finger_third_joint", "_pinky_finger2"),
    ("_pinky_finger2", "_pinky_finger3"), ("_pinky_finger3", "_pinky_finger4"),
]

RIGHT_HAND_LIMBS_NAMES = [(f"right{a}", f"right{b}") for a, b in _HAND_FINGER_LIMBS]
LEFT_HAND_LIMBS_NAMES = [(f"left{a}", f"left{b}") for a, b in _HAND_FINGER_LIMBS]


# ──────────────────────────────────────────────────────────────────────
# PoseHeader components
# ──────────────────────────────────────────────────────────────────────

def sapiens_components():
    return [
        PoseHeaderComponent(
            name="BODY", points=BODY_POINTS,
            limbs=_map_limbs(BODY_POINTS, BODY_LIMBS_NAMES),
            colors=[(0, 255, 0)], point_format="XYC",
        ),
        PoseHeaderComponent(
            name="RIGHT_HAND", points=RIGHT_HAND_POINTS,
            limbs=_map_limbs(RIGHT_HAND_POINTS, RIGHT_HAND_LIMBS_NAMES),
            colors=[(255, 128, 0)], point_format="XYC",
        ),
        PoseHeaderComponent(
            name="LEFT_HAND", points=LEFT_HAND_POINTS,
            limbs=_map_limbs(LEFT_HAND_POINTS, LEFT_HAND_LIMBS_NAMES),
            colors=[(0, 255, 0)], point_format="XYC",
        ),
        PoseHeaderComponent(
            name="BODY_EXTRA", points=BODY_EXTRA_POINTS,
            limbs=[], colors=[(0, 255, 0)], point_format="XYC",
        ),
        PoseHeaderComponent(
            name="FACE", points=FACE_POINTS,
            limbs=[], colors=[(255, 255, 255)], point_format="XYC",
        ),
        PoseHeaderComponent(
            name="LEFT_EAR", points=LEFT_EAR_POINTS,
            limbs=[], colors=[(255, 255, 255)], point_format="XYC",
        ),
        PoseHeaderComponent(
            name="RIGHT_EAR", points=RIGHT_EAR_POINTS,
            limbs=[], colors=[(255, 255, 255)], point_format="XYC",
        ),
        PoseHeaderComponent(
            name="LEFT_IRIS", points=LEFT_IRIS_POINTS,
            limbs=[], colors=[(255, 255, 255)], point_format="XYC",
        ),
        PoseHeaderComponent(
            name="RIGHT_IRIS", points=RIGHT_IRIS_POINTS,
            limbs=[], colors=[(255, 255, 255)], point_format="XYC",
        ),
        PoseHeaderComponent(
            name="LEFT_PUPIL", points=LEFT_PUPIL_POINTS,
            limbs=[], colors=[(255, 255, 255)], point_format="XYC",
        ),
        PoseHeaderComponent(
            name="RIGHT_PUPIL", points=RIGHT_PUPIL_POINTS,
            limbs=[], colors=[(255, 255, 255)], point_format="XYC",
        ),
    ]


# ──────────────────────────────────────────────────────────────────────
# Model download
# ──────────────────────────────────────────────────────────────────────

MODEL_FILENAME = "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"
MODEL_REPO_ID = "facebook/sapiens-pose-1b-torchscript"


def _download_model():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    return path


# ──────────────────────────────────────────────────────────────────────
# Preprocessing & postprocessing
# ──────────────────────────────────────────────────────────────────────

INPUT_SIZE = (1024, 768)  # (height, width) expected by the model


def _create_preprocessor():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ])


def _postprocess_heatmaps(heatmaps: torch.Tensor, img_height: int, img_width: int):
    """Convert model heatmap output to (x, y) keypoints and confidence scores.

    Parameters
    ----------
    heatmaps : torch.Tensor
        Raw model output, shape (batch, num_keypoints, hm_h, hm_w).
    img_height : int
        Original image height.
    img_width : int
        Original image width.

    Returns
    -------
    keypoints : np.ndarray, shape (num_keypoints, 2)
    confidence : np.ndarray, shape (num_keypoints,)
    """
    result = heatmaps[0].detach().cpu()  # (num_keypoints, hm_h, hm_w)

    # Upsample heatmaps to original image resolution
    upsampled = F.interpolate(
        result.unsqueeze(0),
        size=(img_height, img_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # (num_keypoints, img_h, img_w)

    num_kp = upsampled.shape[0]
    flat = upsampled.view(num_kp, -1)  # (num_keypoints, img_h * img_w)

    confidence, flat_indices = flat.max(dim=1)
    ys = (flat_indices // img_width).float()
    xs = (flat_indices % img_width).float()

    keypoints = np.stack([xs.numpy(), ys.numpy()], axis=-1)  # (num_keypoints, 2)
    confidence = confidence.float().numpy()                    # (num_keypoints,)

    return keypoints, confidence


# ──────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────

def process_sapiens(frames, fps: float, use_cpu: bool) -> NumPyPoseBody:
    """Run Sapiens pose estimation on a sequence of RGB frames.

    Parameters
    ----------
    frames : iterable of np.ndarray
        Video frames in RGB format (H, W, 3).
    fps : float
        Frames per second of the video.
    use_cpu : bool
        Force CPU inference.

    Returns
    -------
    NumPyPoseBody
    """
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
    dtype = torch.float32 if device.type == "cpu" else torch.float16

    print("Downloading Sapiens pose model (if not cached)...")
    model_path = _download_model()

    print(f"Loading Sapiens model on {device} ({dtype})...")
    model = torch.jit.load(model_path)
    model = model.eval().to(device).to(dtype)

    preprocessor = _create_preprocessor()

    frames_data = []
    frames_conf = []

    for frame in tqdm(frames, desc="Sapiens pose estimation"):
        # frame is RGB; preprocessor converts via ToPILImage (expects RGB or HWC uint8)
        tensor = preprocessor(frame).to(device).to(dtype)

        with torch.inference_mode():
            heatmaps = model(tensor)

        img_h, img_w = frame.shape[:2]
        keypoints, confidence = _postprocess_heatmaps(heatmaps, img_h, img_w)

        frames_data.append([keypoints])      # (1, 308, 2)
        frames_conf.append([confidence])     # (1, 308)

    data_array = np.array(frames_data, dtype=np.float32)   # (frames, 1, 308, 2)
    conf_array = np.array(frames_conf, dtype=np.float32)   # (frames, 1, 308)

    return NumPyPoseBody(fps=fps, data=data_array, confidence=conf_array)


def estimate_and_load_sapiens(frames,
                              fps: float = 24,
                              use_cpu: bool = False,
                              width: int = 1000,
                              height: int = 1000) -> Pose:
    """Estimate pose with Sapiens and return a Pose object.

    Parameters
    ----------
    frames : iterable of np.ndarray
        Video frames in RGB format.
    fps : float
        Frames per second.
    use_cpu : bool
        Force CPU inference.
    width : int
        Video width (used in header dimensions).
    height : int
        Video height (used in header dimensions).

    Returns
    -------
    Pose
    """
    print("Loading pose with Sapiens...")

    dimensions = PoseHeaderDimensions(width=width, height=height)
    header = PoseHeader(version=0.1,
                        dimensions=dimensions,
                        components=sapiens_components())
    body = process_sapiens(frames, fps, use_cpu)

    return Pose(header, body)
