import logging
import time
from typing import Any, Dict, List

import imageio
import numpy as np

import pathlib
import sys
import os
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ManiFlow'))
sys.path.append(os.path.join(ROOT_DIR, 'ManiFlow', 'ManiFlow'))
sys.path.append(os.path.join(ROOT_DIR, 'ManiFlow', 'ManiFlow', 'maniflow'))
from middleware.config_loader import get_config, get_evo1_config, get_maniflow_config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("datacenter_utils")


def now_ros_stamp():
    ns = time.time_ns()
    return ns // 1_000_000_000, ns % 1_000_000_000


def encode_image_array(img: np.ndarray) -> List:
    """将 uint8 图像转成可 JSON 序列化的嵌套 list。"""
    if img is None:
        return []
    return img.astype(np.uint8).tolist()


def datacenter_obs_to_evo1(obs: Dict[str, Any], prompt: str = "do something.", resize_size: int = 448, arm_dofs: int = 7) -> Dict[str, Any]:
    """将 datacenter 观测转为发给策略的 JSON。"""
    if "image" not in obs:
        return {}

    config = get_config()
    evo1_config = get_evo1_config()
    camera_sequence = config.get_ava_cameras()

    # 构建相机名称到索引的映射
    cam_name_to_idx = {cam.name: idx for idx, cam in enumerate(camera_sequence)}
    camera_images: List[List] = [None] * evo1_config.image_mask_len
    image_mask: List[int] = [0] * evo1_config.image_mask_len

    # 有序构建图像列表和 mask
    for cam_name, cam_data in obs["image"].items():
        if cam_name not in cam_name_to_idx:
            continue
        idx = cam_name_to_idx[cam_name]
        rgb = cam_data.get("rgb_img")
        if rgb is None:
            rgb = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)  # 占位黑图
        else:
            image_mask[idx] = 1
        camera_images[idx] = encode_image_array(rgb)

    # 填充未收到数据的相机位置
    for i, data in enumerate(camera_images):
        if data is None:
            camera_images[i] = encode_image_array(np.zeros((resize_size, resize_size, 3), dtype=np.uint8))

    # 固定长度 mask
    IMAGE_MASK_LEN = evo1_config.image_mask_len
    ACTION_MASK_LEN = evo1_config.action_mask_len
    image_mask = (image_mask + [0] * IMAGE_MASK_LEN)[:IMAGE_MASK_LEN]
    action_mask = ([1] * (arm_dofs) + [0] * ACTION_MASK_LEN)[:ACTION_MASK_LEN]

    return {
        "image": camera_images,
        "state": obs.get("qpos", np.array([])).tolist(),
        "prompt": prompt,
        "image_mask": image_mask,
        "action_mask": action_mask,
    }

"""
FOCUS OBSERVATION DICT STRUCTURE:
camera_name = [front, wrist_right, wrist_left, ...]
{
    image: { camera_name: {rgb_img: np.ndarray, depth_img: np.ndarray, }, ...},
    qpos: np.ndarray,
    prompt: str,
}
"""

def datacenter_obs_to_maniflow(obs: Dict[str, Any], prompt: str = "do something.", resize_size: int = 224, arm_dofs: int = 7) -> Dict[str, Any]:
    if "image" not in obs:
        return {}

    config = get_config()
    camera_sequence = config.get_ava_cameras()

    head_cam = None
    for cam_name, cam_data in obs["image"].items():
        if cam_name == "front":
            rgb = cam_data.get("rgb_img")
            if rgb is None:
                rgb = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)
            head_cam = (np.moveaxis(rgb, -1, 0) / 255.0).astype(np.float32)
            break

    if head_cam is None:
        logger.warning("未找到front相机，使用默认黑图")
        head_cam = np.zeros((3, resize_size, resize_size), dtype=np.float32)

    agent_pos = obs.get("qpos", np.array([])).astype(np.float32)
    # print(f"agent_pos is {agent_pos}")
    # input("Press Enter to continue...")

    return {
        "head_cam": head_cam,
        "agent_pos": agent_pos,
        # "prompt": prompt,
    }

def save_video(frames, video_name, fps, save_dir):
    """保存视频到指定路径。"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{video_name}.mp4")
    if not frames:
        logger.error(f"警告：保存视频 {video_name} 时，frames 为空！")
        return
    imageio.mimsave(path, frames, fps=fps, codec="libx264")  # 显式指定编码器避免默认失败
    logger.info(f"视频已保存到：{path}")
