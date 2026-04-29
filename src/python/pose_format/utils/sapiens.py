import json
import numpy as np
import cv2
import torch
from tqdm import tqdm

from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions

try:
    from sapiens_inference.pose_classes_and_palettes import GOLIATH_KEYPOINTS, GOLIATH_SKELETON_INFO
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "git+https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference.git"])
    from sapiens_inference.pose_classes_and_palettes import GOLIATH_KEYPOINTS, GOLIATH_SKELETON_INFO

EXTENDED_GOLIATH_KEYPOINTS = (GOLIATH_KEYPOINTS + ["left_wrist_body", "right_wrist_body"])

BODY_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear",
    "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle", "left_big_toe",
    "left_small_toe", "left_heel", "right_big_toe", "right_small_toe",
    "right_heel", "left_olecranon", "right_olecranon", "left_cubital_fossa",
    "right_cubital_fossa", "left_acromion", "right_acromion", "neck",
    "left_wrist_body", "right_wrist_body",
]

LEFT_HAND_KEYPOINTS = [
    "left_thumb4", "left_thumb3", "left_thumb2", "left_thumb_third_joint",
    "left_forefinger4", "left_forefinger3", "left_forefinger2", "left_forefinger_third_joint",
    "left_middle_finger4", "left_middle_finger3", "left_middle_finger2", "left_middle_finger_third_joint",
    "left_ring_finger4", "left_ring_finger3", "left_ring_finger2", "left_ring_finger_third_joint",
    "left_pinky_finger4", "left_pinky_finger3", "left_pinky_finger2", "left_pinky_finger_third_joint",
    "left_wrist",
]

RIGHT_HAND_KEYPOINTS = [
    "right_thumb4", "right_thumb3", "right_thumb2", "right_thumb_third_joint",
    "right_forefinger4", "right_forefinger3", "right_forefinger2", "right_forefinger_third_joint",
    "right_middle_finger4", "right_middle_finger3", "right_middle_finger2", "right_middle_finger_third_joint",
    "right_ring_finger4", "right_ring_finger3", "right_ring_finger2", "right_ring_finger_third_joint",
    "right_pinky_finger4", "right_pinky_finger3", "right_pinky_finger2", "right_pinky_finger_third_joint",
    "right_wrist",
]

FACE_KEYPOINTS = [
    "center_of_glabella", "center_of_nose_root", "tip_of_nose_bridge", "midpoint_1_of_nose_bridge",
    "midpoint_2_of_nose_bridge", "midpoint_3_of_nose_bridge", "center_of_labiomental_groove", "tip_of_chin",
    "upper_startpoint_of_r_eyebrow", "lower_startpoint_of_r_eyebrow", "end_of_r_eyebrow", "upper_midpoint_1_of_r_eyebrow",
    "lower_midpoint_1_of_r_eyebrow", "upper_midpoint_2_of_r_eyebrow", "upper_midpoint_3_of_r_eyebrow", "lower_midpoint_2_of_r_eyebrow",
    "lower_midpoint_3_of_r_eyebrow", "upper_startpoint_of_l_eyebrow", "lower_startpoint_of_l_eyebrow", "end_of_l_eyebrow",
    "upper_midpoint_1_of_l_eyebrow", "lower_midpoint_1_of_l_eyebrow", "upper_midpoint_2_of_l_eyebrow", "upper_midpoint_3_of_l_eyebrow",
    "lower_midpoint_2_of_l_eyebrow", "lower_midpoint_3_of_l_eyebrow", "l_inner_end_of_upper_lash_line", "l_outer_end_of_upper_lash_line",
    "l_centerpoint_of_upper_lash_line", "l_midpoint_2_of_upper_lash_line", "l_midpoint_1_of_upper_lash_line", "l_midpoint_6_of_upper_lash_line",
    "l_midpoint_5_of_upper_lash_line", "l_midpoint_4_of_upper_lash_line", "l_midpoint_3_of_upper_lash_line", "l_outer_end_of_upper_eyelid_line",
    "l_midpoint_6_of_upper_eyelid_line", "l_midpoint_2_of_upper_eyelid_line", "l_midpoint_5_of_upper_eyelid_line", "l_centerpoint_of_upper_eyelid_line",
    "l_midpoint_4_of_upper_eyelid_line", "l_midpoint_1_of_upper_eyelid_line", "l_midpoint_3_of_upper_eyelid_line", "l_midpoint_6_of_upper_crease_line",
    "l_midpoint_2_of_upper_crease_line", "l_midpoint_5_of_upper_crease_line", "l_centerpoint_of_upper_crease_line", "l_midpoint_4_of_upper_crease_line",
    "l_midpoint_1_of_upper_crease_line", "l_midpoint_3_of_upper_crease_line", "r_inner_end_of_upper_lash_line", "r_outer_end_of_upper_lash_line",
    "r_centerpoint_of_upper_lash_line", "r_midpoint_1_of_upper_lash_line", "r_midpoint_2_of_upper_lash_line", "r_midpoint_3_of_upper_lash_line",
    "r_midpoint_4_of_upper_lash_line", "r_midpoint_5_of_upper_lash_line", "r_midpoint_6_of_upper_lash_line", "r_outer_end_of_upper_eyelid_line",
    "r_midpoint_3_of_upper_eyelid_line", "r_midpoint_1_of_upper_eyelid_line", "r_midpoint_4_of_upper_eyelid_line", "r_centerpoint_of_upper_eyelid_line",
    "r_midpoint_5_of_upper_eyelid_line", "r_midpoint_2_of_upper_eyelid_line", "r_midpoint_6_of_upper_eyelid_line", "r_midpoint_3_of_upper_crease_line",
    "r_midpoint_1_of_upper_crease_line", "r_midpoint_4_of_upper_crease_line", "r_centerpoint_of_upper_crease_line", "r_midpoint_5_of_upper_crease_line",
    "r_midpoint_2_of_upper_crease_line", "r_midpoint_6_of_upper_crease_line", "l_inner_end_of_lower_lash_line", "l_outer_end_of_lower_lash_line",
    "l_centerpoint_of_lower_lash_line", "l_midpoint_2_of_lower_lash_line", "l_midpoint_1_of_lower_lash_line", "l_midpoint_6_of_lower_lash_line",
    "l_midpoint_5_of_lower_lash_line", "l_midpoint_4_of_lower_lash_line", "l_midpoint_3_of_lower_lash_line", "l_outer_end_of_lower_eyelid_line",
    "l_midpoint_6_of_lower_eyelid_line", "l_midpoint_2_of_lower_eyelid_line", "l_midpoint_5_of_lower_eyelid_line", "l_centerpoint_of_lower_eyelid_line",
    "l_midpoint_4_of_lower_eyelid_line", "l_midpoint_1_of_lower_eyelid_line", "l_midpoint_3_of_lower_eyelid_line", "r_inner_end_of_lower_lash_line",
    "r_outer_end_of_lower_lash_line", "r_centerpoint_of_lower_lash_line", "r_midpoint_1_of_lower_lash_line", "r_midpoint_2_of_lower_lash_line",
    "r_midpoint_3_of_lower_lash_line", "r_midpoint_4_of_lower_lash_line", "r_midpoint_5_of_lower_lash_line", "r_midpoint_6_of_lower_lash_line",
    "r_outer_end_of_lower_eyelid_line", "r_midpoint_3_of_lower_eyelid_line", "r_midpoint_1_of_lower_eyelid_line", "r_midpoint_4_of_lower_eyelid_line",
    "r_centerpoint_of_lower_eyelid_line", "r_midpoint_5_of_lower_eyelid_line", "r_midpoint_2_of_lower_eyelid_line", "r_midpoint_6_of_lower_eyelid_line",
    "tip_of_nose", "bottom_center_of_nose", "r_outer_corner_of_nose", "l_outer_corner_of_nose",
    "inner_corner_of_r_nostril", "outer_corner_of_r_nostril", "upper_corner_of_r_nostril", "inner_corner_of_l_nostril",
    "outer_corner_of_l_nostril", "upper_corner_of_l_nostril", "r_outer_corner_of_mouth", "l_outer_corner_of_mouth",
    "center_of_cupid_bow", "center_of_lower_outer_lip", "midpoint_1_of_upper_outer_lip", "midpoint_2_of_upper_outer_lip",
    "midpoint_1_of_lower_outer_lip", "midpoint_2_of_lower_outer_lip", "midpoint_3_of_upper_outer_lip", "midpoint_4_of_upper_outer_lip",
    "midpoint_5_of_upper_outer_lip", "midpoint_6_of_upper_outer_lip", "midpoint_3_of_lower_outer_lip", "midpoint_4_of_lower_outer_lip",
    "midpoint_5_of_lower_outer_lip", "midpoint_6_of_lower_outer_lip", "r_inner_corner_of_mouth", "l_inner_corner_of_mouth",
    "center_of_upper_inner_lip", "center_of_lower_inner_lip", "midpoint_1_of_upper_inner_lip", "midpoint_2_of_upper_inner_lip",
    "midpoint_1_of_lower_inner_lip", "midpoint_2_of_lower_inner_lip", "midpoint_3_of_upper_inner_lip", "midpoint_4_of_upper_inner_lip",
    "midpoint_5_of_upper_inner_lip", "midpoint_6_of_upper_inner_lip", "midpoint_3_of_lower_inner_lip", "midpoint_4_of_lower_inner_lip",
    "midpoint_5_of_lower_inner_lip", "midpoint_6_of_lower_inner_lip", "l_top_end_of_inferior_crus", "l_top_end_of_superior_crus",
    "l_start_of_antihelix", "l_end_of_antihelix", "l_midpoint_1_of_antihelix", "l_midpoint_1_of_inferior_crus",
    "l_midpoint_2_of_antihelix", "l_midpoint_3_of_antihelix", "l_point_1_of_inner_helix", "l_point_2_of_inner_helix",
    "l_point_3_of_inner_helix", "l_point_4_of_inner_helix", "l_point_5_of_inner_helix", "l_point_6_of_inner_helix",
    "l_point_7_of_inner_helix", "l_highest_point_of_antitragus", "l_bottom_point_of_tragus", "l_protruding_point_of_tragus",
    "l_top_point_of_tragus", "l_start_point_of_crus_of_helix", "l_deepest_point_of_concha", "l_tip_of_ear_lobe",
    "l_midpoint_between_22_15", "l_bottom_connecting_point_of_ear_lobe", "l_top_connecting_point_of_helix", "l_point_8_of_inner_helix",
    "r_top_end_of_inferior_crus", "r_top_end_of_superior_crus", "r_start_of_antihelix", "r_end_of_antihelix",
    "r_midpoint_1_of_antihelix", "r_midpoint_1_of_inferior_crus", "r_midpoint_2_of_antihelix", "r_midpoint_3_of_antihelix",
    "r_point_1_of_inner_helix", "r_point_8_of_inner_helix", "r_point_3_of_inner_helix", "r_point_4_of_inner_helix",
    "r_point_5_of_inner_helix", "r_point_6_of_inner_helix", "r_point_7_of_inner_helix", "r_highest_point_of_antitragus",
    "r_bottom_point_of_tragus", "r_protruding_point_of_tragus", "r_top_point_of_tragus", "r_start_point_of_crus_of_helix",
    "r_deepest_point_of_concha", "r_tip_of_ear_lobe", "r_midpoint_between_22_15", "r_bottom_connecting_point_of_ear_lobe",
    "r_top_connecting_point_of_helix", "r_point_2_of_inner_helix", "l_center_of_iris", "l_border_of_iris_3",
    "l_border_of_iris_midpoint_1", "l_border_of_iris_12", "l_border_of_iris_midpoint_4", "l_border_of_iris_9",
    "l_border_of_iris_midpoint_3", "l_border_of_iris_6", "l_border_of_iris_midpoint_2", "r_center_of_iris",
    "r_border_of_iris_3", "r_border_of_iris_midpoint_1", "r_border_of_iris_12", "r_border_of_iris_midpoint_4",
    "r_border_of_iris_9", "r_border_of_iris_midpoint_3", "r_border_of_iris_6", "r_border_of_iris_midpoint_2",
    "l_center_of_pupil", "l_border_of_pupil_3", "l_border_of_pupil_midpoint_1", "l_border_of_pupil_12",
    "l_border_of_pupil_midpoint_4", "l_border_of_pupil_9", "l_border_of_pupil_midpoint_3", "l_border_of_pupil_6",
    "l_border_of_pupil_midpoint_2", "r_center_of_pupil", "r_border_of_pupil_3", "r_border_of_pupil_midpoint_1",
    "r_border_of_pupil_12", "r_border_of_pupil_midpoint_4", "r_border_of_pupil_9", "r_border_of_pupil_midpoint_3",
    "r_border_of_pupil_6", "r_border_of_pupil_midpoint_2",
]

BODY_LIMBS_NAMES = [
    ("left_ankle", "left_knee"), ("left_knee", "left_hip"), ("right_ankle", "right_knee"), ("right_knee", "right_hip"),
    ("left_hip", "right_hip"), ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist_body"), ("right_elbow", "right_wrist_body"),
    ("left_eye", "right_eye"), ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"), ("left_ankle", "left_big_toe"), ("left_ankle", "left_small_toe"),
    ("left_ankle", "left_heel"), ("right_ankle", "right_big_toe"), ("right_ankle", "right_small_toe"), ("right_ankle", "right_heel"),
]

LEFT_HAND_LIMBS_NAMES = [
    ("left_wrist", "left_thumb_third_joint"), ("left_thumb_third_joint", "left_thumb2"), ("left_thumb2", "left_thumb3"), ("left_thumb3", "left_thumb4"),
    ("left_wrist", "left_forefinger_third_joint"), ("left_forefinger_third_joint", "left_forefinger2"), ("left_forefinger2", "left_forefinger3"), ("left_forefinger3", "left_forefinger4"),
    ("left_wrist", "left_middle_finger_third_joint"), ("left_middle_finger_third_joint", "left_middle_finger2"), ("left_middle_finger2", "left_middle_finger3"), ("left_middle_finger3", "left_middle_finger4"),
    ("left_wrist", "left_ring_finger_third_joint"), ("left_ring_finger_third_joint", "left_ring_finger2"), ("left_ring_finger2", "left_ring_finger3"), ("left_ring_finger3", "left_ring_finger4"),
    ("left_wrist", "left_pinky_finger_third_joint"), ("left_pinky_finger_third_joint", "left_pinky_finger2"), ("left_pinky_finger2", "left_pinky_finger3"), ("left_pinky_finger3", "left_pinky_finger4"),
]

RIGHT_HAND_LIMBS_NAMES = [
    ("right_wrist", "right_thumb_third_joint"), ("right_thumb_third_joint", "right_thumb2"), ("right_thumb2", "right_thumb3"), ("right_thumb3", "right_thumb4"),
    ("right_wrist", "right_forefinger_third_joint"), ("right_forefinger_third_joint", "right_forefinger2"), ("right_forefinger2", "right_forefinger3"), ("right_forefinger3", "right_forefinger4"),
    ("right_wrist", "right_middle_finger_third_joint"), ("right_middle_finger_third_joint", "right_middle_finger2"), ("right_middle_finger2", "right_middle_finger3"), ("right_middle_finger3", "right_middle_finger4"),
    ("right_wrist", "right_ring_finger_third_joint"), ("right_ring_finger_third_joint", "right_ring_finger2"), ("right_ring_finger2", "right_ring_finger3"), ("right_ring_finger3", "right_ring_finger4"),
    ("right_wrist", "right_pinky_finger_third_joint"), ("right_pinky_finger_third_joint", "right_pinky_finger2"), ("right_pinky_finger2", "right_pinky_finger3"), ("right_pinky_finger3", "right_pinky_finger4"),
]

GENERIC_HAND_KEYPOINTS = [element.replace("right_", "") for element in RIGHT_HAND_KEYPOINTS]

name_to_global = {name: i for i, name in enumerate(EXTENDED_GOLIATH_KEYPOINTS)}

LEFT_WRIST_IDX = name_to_global["left_wrist"]
RIGHT_WRIST_IDX = name_to_global["right_wrist"]

LEFT_WRIST_BODY_IDX = name_to_global["left_wrist_body"]
RIGHT_WRIST_BODY_IDX = name_to_global["right_wrist_body"]


# ──────────────────────────────────────────────────────────────────────
# PoseHeader components
# ──────────────────────────────────────────────────────────────────────

def get_sapiens_components():
    """
    Create PoseHeader components for Sapiens (Goliath),
    split into BODY_SAPIENS / FACE_SAPIENS / LEFT_HAND_SAPIENS / RIGHT_HAND_SAPIENS.
    """

    def build_limbs(keypoint_names, named_limbs):
        global_indices = [name_to_global[k] for k in keypoint_names]
        global_to_local = {g: i for i, g in enumerate(global_indices)}
        limbs = []
        for link in named_limbs:
            a, b = link[0], link[1]
            if a in name_to_global and b in name_to_global:
                ga, gb = name_to_global[a], name_to_global[b]
                if ga in global_to_local and gb in global_to_local:
                    limbs.append((global_to_local[ga], global_to_local[gb]))
        return limbs

    body_limbs = build_limbs(BODY_KEYPOINTS, BODY_LIMBS_NAMES)
    l_hand_limbs = build_limbs(LEFT_HAND_KEYPOINTS, LEFT_HAND_LIMBS_NAMES)
    r_hand_limbs = build_limbs(RIGHT_HAND_KEYPOINTS, RIGHT_HAND_LIMBS_NAMES)

    return [
        PoseHeaderComponent(
            name="BODY_SAPIENS",
            points=BODY_KEYPOINTS,
            limbs=body_limbs,
            colors=[(0, 255, 0)] * len(body_limbs),
            point_format="XYC",
        ),
        PoseHeaderComponent(
            name="FACE_SAPIENS",
            points=FACE_KEYPOINTS,
            limbs=[],
            colors=[(255, 255, 255)],
            point_format="XYC",
        ),
        PoseHeaderComponent(
            name="LEFT_HAND_SAPIENS",
            points=GENERIC_HAND_KEYPOINTS,
            limbs=l_hand_limbs,
            colors=[(255, 0, 0)] * len(l_hand_limbs),
            point_format="XYC",
        ),
        PoseHeaderComponent(
            name="RIGHT_HAND_SAPIENS",
            points=GENERIC_HAND_KEYPOINTS,
            limbs=r_hand_limbs,
            colors=[(0, 0, 255)] * len(r_hand_limbs),
            point_format="XYC",
        ),
    ]


# ──────────────────────────────────────────────────────────────────────
# JSON → Pose conversion
# ──────────────────────────────────────────────────────────────────────

def load_sapiens_json(json_path):
    """
    Load a Sapiens (Goliath) pose JSON file.

    Expected JSON structure:
        {
          "metadata": {"fps": float, "width": int, "height": int, "num_keypoints": int},
          "frames": [{"frame": int, "keypoints": {"<name>": [x, y, score], ...}}, ...]
        }

    Returns
    -------
    frames : list of dict, sorted by frame index
    metadata : dict
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    assert isinstance(raw, dict) and "frames" in raw, "Invalid Sapiens JSON format"

    frames = raw["frames"]
    metadata = raw.get("metadata", {})
    frames = sorted(frames, key=lambda x: x.get("frame", 0))
    return frames, metadata


def parse_sapiens_frame(frame, keypoint_names):
    """Parse a single Sapiens frame dict into (xy, conf) arrays in the order of keypoint_names."""
    kpts = frame["keypoints"]
    K = len(keypoint_names)

    xy = np.zeros((K, 2), dtype=np.float32)
    conf = np.zeros((K,), dtype=np.float32)

    for i, name in enumerate(keypoint_names):
        if name in kpts:
            x, y, score = kpts[name]
            xy[i, 0] = x
            xy[i, 1] = y
            conf[i] = score

    return xy, conf


def duplicate_wrists(xy, conf):
    """
    Extend xy/conf by duplicating wrist keypoints into dedicated BODY wrist entries
    (left_wrist_body, right_wrist_body). The LEFT_HAND/RIGHT_HAND components keep
    the original wrists as hand roots; the BODY component gets its own wrist joints.
    """
    K = xy.shape[0]

    xy_ext = np.zeros((K + 2, 2), dtype=xy.dtype)
    conf_ext = np.zeros((K + 2,), dtype=conf.dtype)

    xy_ext[:K] = xy
    conf_ext[:K] = conf

    xy_ext[LEFT_WRIST_BODY_IDX] = xy[LEFT_WRIST_IDX]
    xy_ext[RIGHT_WRIST_BODY_IDX] = xy[RIGHT_WRIST_IDX]

    conf_ext[LEFT_WRIST_BODY_IDX] = conf[LEFT_WRIST_IDX]
    conf_ext[RIGHT_WRIST_BODY_IDX] = conf[RIGHT_WRIST_IDX]

    return xy_ext, conf_ext


def reorder_sapiens_keypoints(xy, conf):
    """
    Reorder from EXTENDED_GOLIATH order to [BODY, FACE, LEFT_HAND, RIGHT_HAND].
    """
    reordered_names = (
        BODY_KEYPOINTS
        + FACE_KEYPOINTS
        + LEFT_HAND_KEYPOINTS
        + RIGHT_HAND_KEYPOINTS
    )

    name_to_idx = {name: i for i, name in enumerate(EXTENDED_GOLIATH_KEYPOINTS)}

    if set(reordered_names) != set(EXTENDED_GOLIATH_KEYPOINTS):
        missing = set(EXTENDED_GOLIATH_KEYPOINTS) - set(reordered_names)
        extra = set(reordered_names) - set(EXTENDED_GOLIATH_KEYPOINTS)
        raise ValueError(f"Keypoint mismatch.\nMissing: {missing}\nExtra: {extra}")

    reorder_indices = np.array([name_to_idx[name] for name in reordered_names], dtype=np.int64)

    return xy[reorder_indices], conf[reorder_indices]


def _sapiens_json_to_pose(frames, fps, width, height, version=0.1, depth=0) -> Pose:
    """Build a Pose from a list of frame dicts (each with 'frame' and 'keypoints')."""
    frames_xy = []
    frames_conf = []

    for frame in frames:
        xy, conf = parse_sapiens_frame(frame, GOLIATH_KEYPOINTS)
        xy, conf = duplicate_wrists(xy, conf)
        xy, conf = reorder_sapiens_keypoints(xy, conf)
        frames_xy.append(xy)
        frames_conf.append(conf)

    xy_data = np.stack(frames_xy, axis=0)[:, None, :, :]   # (T, 1, K, 2)
    conf_data = np.stack(frames_conf, axis=0)[:, None, :]  # (T, 1, K)

    header = PoseHeader(
        version=version,
        dimensions=PoseHeaderDimensions(width=width, height=height, depth=depth),
        components=get_sapiens_components(),
    )
    body = NumPyPoseBody(fps=fps, data=xy_data, confidence=conf_data)
    return Pose(header, body)


def load_sapiens_wholebody_from_json(
    input_path: str,
    version: float = 0.1,
    fps: float = 24,
    width: int = 1000,
    height: int = 1000,
    depth: int = 0,
) -> Pose:
    """Load Sapiens Goliath poses from a JSON file and return a Pose."""
    print("Loading pose with Sapiens (Goliath)...")

    frames, metadata = load_sapiens_json(input_path)

    fps = metadata.get("fps", fps)
    width = metadata.get("width", width)
    height = metadata.get("height", height)

    num_keypoints = metadata.get("num_keypoints", len(GOLIATH_KEYPOINTS))
    assert num_keypoints == len(GOLIATH_KEYPOINTS), "Mismatch between JSON and GOLIATH_KEYPOINTS"

    return _sapiens_json_to_pose(frames, fps=fps, width=width, height=height, version=version, depth=depth)


# ──────────────────────────────────────────────────────────────────────
# Direct video inference (produces the same JSON frame structure in memory
# and reuses the JSON → Pose helpers).
# Requires the `sapiens_inference` package from Sapiens-Pytorch-Inference.
# ──────────────────────────────────────────────────────────────────────

# HuggingFace's "main" branch of facebook/sapiens-pose-1b-torchscript currently
# 404s on the .pt2 file, so pin to a known-good commit (matches install_sapiens.sh
# in multimodalhugs-pipelines).
_SAPIENS_HF_REVISION = "4caa2b2290255dc8963b5ead35fe3c6e761742aa"
_SAPIENS_POSE_FILE = "sapiens_1b_goliath_best_goliath_AP_640_torchscript.pt2"
_SAPIENS_POSE_REPO = "facebook/sapiens-pose-1b-torchscript"


def _pin_sapiens_hf_revision():
    """
    Force any hf_hub_download call for facebook/sapiens-* to use the pinned revision.
    Patches both huggingface_hub.hf_hub_download (for future imports) and any
    already-imported references inside sapiens_inference.* submodules.
    """
    import sys
    import huggingface_hub

    def _wrap(original):
        def _patched(*args, **kwargs):
            repo_id = kwargs.get("repo_id") or (args[0] if args else "")
            if isinstance(repo_id, str) and repo_id.startswith("facebook/sapiens-"):
                kwargs.setdefault("revision", _SAPIENS_HF_REVISION)
            return original(*args, **kwargs)
        _patched._sapiens_pinned = True
        return _patched

    if not getattr(huggingface_hub.hf_hub_download, "_sapiens_pinned", False):
        huggingface_hub.hf_hub_download = _wrap(huggingface_hub.hf_hub_download)

    for name, module in list(sys.modules.items()):
        if not name.startswith("sapiens_inference"):
            continue
        fn = getattr(module, "hf_hub_download", None)
        if fn is not None and not getattr(fn, "_sapiens_pinned", False):
            module.hf_hub_download = _wrap(fn)


def _ensure_sapiens_model_and_cwd():
    """
    Mirrors install_sapiens.sh: download the Sapiens pose model into
    <sapiens_repo>/models/<file> (with the pinned revision) and chdir to the
    repo root so SapiensPoseEstimation's relative model paths resolve.
    """
    import os
    import shutil
    import sapiens_inference

    pkg_path = os.path.dirname(os.path.abspath(sapiens_inference.__file__))
    repo_root = os.path.dirname(pkg_path)
    models_dir = os.path.join(repo_root, "models")
    target = os.path.join(models_dir, _SAPIENS_POSE_FILE)

    if not os.path.isfile(target):
        os.makedirs(models_dir, exist_ok=True)
        print("Downloading Sapiens pose model...")
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(
            repo_id=_SAPIENS_POSE_REPO,
            filename=_SAPIENS_POSE_FILE,
            revision=_SAPIENS_HF_REVISION,
        )
        shutil.copy(src, target)

    if os.getcwd() != repo_root:
        print(f"Changing working directory to: {repo_root}")
        os.chdir(repo_root)


def _lazy_import_sapiens_inference():
    _pin_sapiens_hf_revision()
    try:
        from sapiens_inference.pose import SapiensPoseEstimation, SapiensPoseEstimationType
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "git+https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference.git"])
        from sapiens_inference.pose import SapiensPoseEstimation, SapiensPoseEstimationType
    _pin_sapiens_hf_revision()
    _ensure_sapiens_model_and_cwd()
    return SapiensPoseEstimation, SapiensPoseEstimationType


def _get_device(use_cpu: bool):
    if use_cpu:
        return torch.device("cpu"), torch.float32
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def _sapiens_frames_to_json(frames, use_cpu: bool):
    """Yield per-frame {'frame': idx, 'keypoints': {name: [x, y, score]}} dicts."""
    SapiensPoseEstimation, SapiensPoseEstimationType = _lazy_import_sapiens_inference()

    # Upstream sapiens_inference hardcodes the 1B torchscript filename as AP_640, but
    # Meta retrained the model and the file on HuggingFace is now AP_639. Patch the
    # enum value so download_hf_model constructs a URL that actually resolves.
    # Remove this once https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference is updated.
    SapiensPoseEstimationType.POSE_ESTIMATION_1B._value_ = (
        "sapiens-pose-1b-torchscript/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"
    )

    device, dtype = _get_device(use_cpu)
    print(f"Loading Sapiens model on {device} ({dtype})...")
    estimator = SapiensPoseEstimation(
        type=SapiensPoseEstimationType.POSE_ESTIMATION_1B,
        device=device,
        dtype=dtype,
    )

    for frame_idx, frame_rgb in enumerate(tqdm(frames, desc="Sapiens pose estimation")):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        kpts = {}
        with torch.no_grad():
            bboxes = estimator.detector.detect(frame_bgr)
            if len(bboxes) > 0:
                _, all_keypoints = estimator.estimate_pose(frame_bgr, [bboxes[0]])
                kpts = {
                    name: [float(x), float(y), float(score)]
                    for name, (x, y, score) in all_keypoints[0].items()
                }
        yield {"frame": frame_idx, "keypoints": kpts}


def estimate_and_load_sapiens(frames,
                              fps: float = 24,
                              use_cpu: bool = False,
                              width: int = 1000,
                              height: int = 1000) -> Pose:
    """Estimate pose with Sapiens on RGB frames and return a Pose object."""
    print("Loading pose with Sapiens...")
    frame_entries = list(_sapiens_frames_to_json(frames, use_cpu=use_cpu))
    return _sapiens_json_to_pose(frame_entries, fps=fps, width=width, height=height)
