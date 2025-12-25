#!/usr/bin/env python3


import asyncio
import json
import logging
import os
import queue
import shutil
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import pandas as pd
import websockets
from config_loader import ArmConfig, CameraConfig, get_config, get_evo1_config
from datacenter import InteractionDataCenter
from utils import datacenter_obs_to_evo1, save_video

ARM_TOPICS = []
CAMERA_TOPICS = []


"""
FOCUS OBSERVATION DICT STRUCTURE:
camera_name = [front, wrist_right, wrist_left, ...]
{
    image: { camera_name: {rgb_img: np.ndarray, depth_img: np.ndarray, }, ...},
    qpos: np.ndarray,
    prompt: str,
}
"""
evo1_config = get_evo1_config()

# ---- Logging ----
os.makedirs(evo1_config.log_save_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("maniflow_real_robot_eval")


class BackgroundIOWorker:
    """后台写盘线程 images/CSV/video."""

    def __init__(self, csv_path: str, max_queue: int = 2048):
        self.csv_path = csv_path
        self.q: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue(max_queue)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._stop_evt = threading.Event()
        self._thread.start()

    def submit_image(self, path: str, img: np.ndarray):
        self._put("image", {"path": path, "img": img})

    def submit_csv(self, row: Dict[str, Any]):
        self._put("csv", {"row": row})

    def submit_video(self, frames: List[np.ndarray], video_name: str, fps: int, save_dir: str):
        self._put("video", {"frames": frames, "video_name": video_name, "fps": fps, "save_dir": save_dir})

    def _put(self, kind: str, payload: Dict[str, Any]):
        try:
            self.q.put_nowait((kind, payload))
        except queue.Full:
            logger.warning("IO queue full, dropping task of type %s", kind)

    def _worker(self):
        while not self._stop_evt.is_set():
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            kind, payload = item
            try:
                if kind == "image":
                    imageio.imwrite(payload["path"], payload["img"].astype(np.uint8))
                elif kind == "csv":
                    pd.DataFrame([payload["row"]]).to_csv(self.csv_path, mode="a", header=False, index=False)
                elif kind == "video":
                    save_video(payload["frames"], payload["video_name"], fps=payload["fps"], save_dir=payload["save_dir"])
            except Exception:
                logger.exception("Background IO task failed (%s)", kind)
            finally:
                self.q.task_done()

    def close(self, timeout: Optional[float] = 5.0):
        self._stop_evt.set()
        # Drain remaining tasks quickly
        end_time = time.time() + (timeout or 0)
        while not self.q.empty() and time.time() < end_time:
            try:
                kind, payload = self.q.get_nowait()
            except queue.Empty:
                break
            try:
                if kind == "image":
                    imageio.imwrite(payload["path"], payload["img"].astype(np.uint8))
                elif kind == "csv":
                    pd.DataFrame([payload["row"]]).to_csv(self.csv_path, mode="a", header=False, index=False)
                elif kind == "video":
                    save_video(payload["frames"], payload["video_name"], fps=payload["fps"], save_dir=payload["save_dir"])
            except Exception:
                logger.exception("Background IO drain failed (%s)", kind)
        self._thread.join(timeout=timeout)


class EpisodeLogger:
    """独立的日志记录器，负责所有与Episode相关的日志写盘。"""

    def __init__(self, log_save_dir: str, arms_config: List[ArmConfig], cameras_config: List[CameraConfig], robot_config):
        """初始化日志记录器。

        Args:
            log_save_dir: 日志保存目录
            arms_config: 机械臂配置列表
            cameras_config: 相机配置列表
            robot_config: 机器人全局配置对象
        """
        self.enabled = False
        self.io_worker: Optional[BackgroundIOWorker] = None
        self.csv_cols: List[str] = []

        # 目录路径
        self.log_dir = log_save_dir
        self.csv_path = os.path.join(log_save_dir, "robot_log.csv")
        self.img_dir = os.path.join(log_save_dir, "images")
        self.img_inference_dir = os.path.join(log_save_dir, "inference_images")

        # 配置缓存
        self.arms = arms_config
        self.cameras = cameras_config
        self.config = robot_config

        # 计算总自由度
        self.arm_dofs = sum(arm.dof + 1 for arm in arms_config)

        self._setup()

    def _setup(self):
        """初始化目录和CSV。"""
        # 清理并创建目录
        shutil.rmtree(self.log_dir, ignore_errors=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.img_inference_dir, exist_ok=True)

        # 初始化CSV和后台IO工作线程
        self._init_csv()
        self.io_worker = BackgroundIOWorker(self.csv_path)
        self.enabled = True

    def _init_csv(self):
        """初始化CSV表头。"""
        cols = ["ep", "step", "h", "img_timestamp", "state_timestamp", "image_path_list"]

        # 为每个机械臂生成独立的 state/action 列
        for arm in self.arms:
            for joint_idx in range(arm.dof):
                cols.append(f"{arm.name}_state_joint_{joint_idx}")
            cols.append(f"{arm.name}_state_gripper")

            for joint_idx in range(arm.dof):
                cols.append(f"{arm.name}_action_joint_{joint_idx}")
            cols.append(f"{arm.name}_action_gripper")

        # 添加原始时间戳列
        for cam in self.cameras:
            cols.append(f"{cam.name}_orig_ts")
        for arm in self.arms:
            cols.append(f"{arm.name}_orig_ts")

        pd.DataFrame(columns=cols).to_csv(self.csv_path, index=False)
        self.csv_cols = cols

    def save_inference_image(self, img: np.ndarray, cam_name: str, ep_idx: int, step_idx: int) -> None:
        """保存推理用的输入图像。"""
        if not self.enabled:
            return
        img_path = os.path.join(self.img_inference_dir, f"inference_ep{ep_idx + 1}_chunk{step_idx:04d}_{cam_name}.png")
        self.io_worker.submit_image(img_path, img)

    def record_step(
        self,
        ep: int,
        step: int,
        h: int,
        obs: Dict[str, Any],
        action: np.ndarray,
        frame_images: Dict[str, np.ndarray],
        orig_image_ts: Dict[str, float],
        orig_state_ts: Dict[str, float],
    ) -> None:
        """记录单个步骤（包括图像、CSV行）。

        Args:
            ep: Episode索引（0-based）
            step: 整个Episode内的步数
            h: 当前horizon步数
            obs: 完整观测字典
            action: 执行的动作向量
            frame_images: 保存的相机图像字典 {cam_name: img}
            orig_image_ts: 各相机的原始时间戳
            orig_state_ts: 各机械臂的原始状态时间戳
        """
        if not self.enabled:
            return

        ts_img = obs.get("image_ts", 0)
        ts_state = obs.get("qpos_ts", 0)
        qpos = obs.get("qpos", np.array([]))

        # 保存图像并收集路径
        image_paths = self._save_frame_images(ep, step, h, frame_images)

        # 构建CSV行
        row = self._build_csv_row(ep, step, h, ts_img, ts_state, image_paths, qpos, action, orig_image_ts, orig_state_ts)

        # 提交CSV写入
        self.io_worker.submit_csv(row)

    def _save_frame_images(self, ep: int, step: int, h: int, frame_images: Dict[str, np.ndarray]) -> List[str]:
        """保存frame_images并返回路径列表。"""
        image_paths = []
        for cam_name, img in frame_images.items():
            img_path = os.path.join(self.img_dir, f"ep{ep + 1}_step{step:04d}_h{h}_{cam_name}.png")
            self.io_worker.submit_image(img_path, img)
            image_paths.append(img_path)
        return image_paths

    def _build_csv_row(
        self,
        ep: int,
        step: int,
        h: int,
        ts_img: float,
        ts_state: float,
        image_paths: List[str],
        qpos: np.ndarray,
        action: np.ndarray,
        orig_image_ts: Dict[str, float],
        orig_state_ts: Dict[str, float],
    ) -> Dict[str, Any]:
        """构建单行CSV数据。"""
        row = {
            "ep": ep + 1,
            "step": step,
            "h": h,
            "img_timestamp": int(ts_img),
            "state_timestamp": int(ts_state),
            "image_path_list": ";".join(image_paths),
        }

        # 按机械臂分别记录 state 和 action
        qpos_offset = 0
        action_offset = 0

        for arm in self.arms:
            arm_len = arm.dof + 1  # joints + gripper

            # 记录 state (qpos)
            for joint_idx in range(arm.dof):
                idx = qpos_offset + joint_idx
                row[f"{arm.name}_state_joint_{joint_idx}"] = float(qpos[idx]) if idx < len(qpos) else 0.0
            row[f"{arm.name}_state_gripper"] = float(qpos[qpos_offset + arm.dof]) if (qpos_offset + arm.dof) < len(qpos) else 0.0

            # 记录 action
            for joint_idx in range(arm.dof):
                idx = action_offset + joint_idx
                row[f"{arm.name}_action_joint_{joint_idx}"] = float(action[idx]) if idx < len(action) else 0.0
            row[f"{arm.name}_action_gripper"] = float(action[action_offset + arm.dof]) if (action_offset + arm.dof) < len(action) else 0.0

            qpos_offset += arm_len
            action_offset += arm_len

        # 记录原始时间戳
        for cam_name, ts in orig_image_ts.items():
            row[f"{cam_name}_orig_ts"] = ts
        for arm_name, ts in orig_state_ts.items():
            row[f"{arm_name}_orig_ts"] = ts

        return row

    def submit_video(self, frames: List[np.ndarray], video_name: str, fps: int) -> None:
        """提交视频写入任务。"""
        if not self.enabled:
            return
        self.io_worker.submit_video(frames, video_name, fps=fps, save_dir=self.log_dir)

    def close(self, timeout: Optional[float] = 5.0) -> None:
        """关闭日志记录器（等待所有待处理任务完成）。"""
        if self.io_worker:
            self.io_worker.close(timeout=timeout)
        self.enabled = False


# ---- Main evaluator class ----
class Evo1RealRobotEvaluator:
    def __init__(self, datacenter: InteractionDataCenter, server_url: str):
        self.datacenter = datacenter
        self.server_url = server_url
        self.horizon = evo1_config.horizon

        self.config = get_config()

        self.arm_dofs = 0  # 包含夹爪的机械臂总自由度
        for arm_config in ARM_TOPICS:
            self.arm_dofs += arm_config.dof + 1

        logger.info(f"总自由度: {self.arm_dofs}")

        assert self.arm_dofs != 0, "机械臂自由度不应为0，请检查"

        self.total_episodes = 0
        self.write_logs = evo1_config.write_logs

        # 使用新的EpisodeLogger
        self.episode_logger: Optional[EpisodeLogger] = None
        if self.write_logs:
            self.episode_logger = EpisodeLogger(
                log_save_dir=evo1_config.log_save_dir,
                arms_config=ARM_TOPICS,
                cameras_config=CAMERA_TOPICS,
                robot_config=self.config,
            )

    async def run_episode(self, ws, ep_idx: int, max_steps: int):
        logger.info(f"正在推理第 {ep_idx + 1} 个Epoch...")
        frames = [] if self.write_logs else None
        steps_taken = 0

        for step_idx in range(max_steps):
            input("等待用户确认请求推理")
            start_inference_time = time.time()
            obs = self.datacenter.get_observation()

            if not obs:
                logger.warning("get_observation returned empty (timeout or no data). Ending episode.")
                break

            # 保存推理用的输入图像
            camera_orig_ts_dict, state_orig_ts_dict = {}, {}
            for cam_config in CAMERA_TOPICS:
                cam_name = cam_config.name
                inference_img = obs.get("image", {}).get(cam_name, {}).get("rgb_img")
                if inference_img is None:
                    inference_img = np.zeros((224, 224, 3), dtype=np.uint8)
                if self.write_logs and self.episode_logger:
                    self.episode_logger.save_inference_image(inference_img, cam_name, ep_idx, step_idx)
                cam_orig_ts = obs.get(f"orig_image_ts_{cam_name}", 0)
                camera_orig_ts_dict[cam_name] = cam_orig_ts

            for arm_config in ARM_TOPICS:
                arm_name = arm_config.name
                state_ts = obs.get(f"orig_qpos_ts_{arm_name}", 0)
                state_orig_ts_dict[arm_name] = state_ts

            # 初次记录时，action 为空数组（长度为所有机械臂的总DOF+夹爪数）
            total_action_len = sum(arm.dof + 1 for arm in ARM_TOPICS)
            empty_action = np.zeros(total_action_len, dtype=float)
            if self.write_logs and self.episode_logger:
                self.episode_logger.record_step(
                    ep=ep_idx,
                    step=steps_taken,
                    h=ep_idx,
                    obs=obs,
                    action=empty_action,
                    frame_images={},
                    orig_image_ts=camera_orig_ts_dict,
                    orig_state_ts=state_orig_ts_dict,
                )

            if not obs:
                logger.warning("get_observation returned empty (timeout or no data). Ending episode.")
                break

            # prepare and send observation
            obs_json = datacenter_obs_to_evo1(obs, arm_dofs=self.arm_dofs)

            try:
                await ws.send(json.dumps(obs_json))
            except Exception as e:
                logger.error(f"Failed to send observation: {e}")
                break
            logger.debug(f"[ep{ep_idx + 1}|step{step_idx}] observation sent")

            # wait for action (with timeout)
            try:
                recv_task = asyncio.create_task(ws.recv())
                done, pending = await asyncio.wait({recv_task}, timeout=evo1_config.action_timeout)
                if not done:
                    logger.warning("Action response timeout.")
                    recv_task.cancel()
                    break
                result = recv_task.result()
                end_inference_time = time.time()

                logger.info(f"[CHUNK TIME][ep{ep_idx + 1}|step{step_idx}] Received action after {end_inference_time - start_inference_time:.2f}s")
            except Exception as e:
                logger.error(f"Failed to receive action: {e}")
                break

            # parse actions
            try:
                action_list = json.loads(result)
                print("==================")
                print(action_list)
                print(len(action_list))

                action_arr = np.array(action_list, dtype=float)
                print("==================")
                print(action_arr)
                print(action_arr.shape)
                input("请检查动作并确认发送")
                # policy might return horizon x action_dim; take first (or per-horizon)
                if action_arr.ndim == 2:
                    # pick the first action (or handle horizon below)
                    # we will execute actions for i in range(min(horizon, action_arr.shape[0]))
                    pass
                elif action_arr.ndim == 1:
                    # (24,) shape, make it (1, 24)
                    action_arr = action_arr[None, :]

            except Exception as e:
                logger.error(f"Action parsing failed: {e}; raw: {result}")
                break

            # Execute up to horizon steps locally (publish to robot)
            for h in range(min(self.horizon, action_arr.shape[0])):
                loop_start = time.time()
                dt = 1 / evo1_config.send_freq
                single_action = action_arr[h]
                if single_action.size < self.arm_dofs:
                    # pad with zeros to match expectation
                    pad_len = self.arm_dofs - single_action.size
                    single_action = np.concatenate([single_action, np.zeros(pad_len, dtype=float)])

                # Publish action (blocking)
                print(f"single_action:{single_action}")
                publish_action = single_action[: self.arm_dofs]
                # print(f"publish_action:{publish_action}")
                # gripper_cmd = single_action[evo1_config.arm_dofs]
                # print(f"gripper_cmd:{gripper_cmd:.9f}")

                # if gripper_cmd <0.9:
                #     publish_action = np.concatenate([publish_action[: evo1_config.arm_dof], [0.0]])
                # else:
                #     publish_action = np.concatenate([publish_action[: evo1_config.arm_dof], [1.0]])
                logger.info(f"发布动作 第{h}步：{publish_action} 总共{action_arr.shape[0]} 步")
                # input("等待用户确认再次发布动作")
                next_obs = self.datacenter.step(publish_action)
                if not next_obs:
                    logger.warning("No observation after action publish; stopping.")
                    break

                cam_frames = {}
                # stack agentview + wrist frames (if available) for saving
                try:
                    # attempt to get two camera views in stable order
                    imgs = []
                    cam_orig_ts_dict = {}
                    state_orig_ts_dict = {}
                    for cam_name, cam_data in next_obs["image"].items():
                        # 从 next_obs 获取当前观测的原始时间戳
                        cam_orig_ts = next_obs.get(f"orig_image_ts_{cam_name}", 0)
                        cam_orig_ts_dict[cam_name] = cam_orig_ts
                        img = cam_data.get("rgb_img")
                        if img is None:
                            img = np.zeros((224, 224, 3), dtype=np.uint8)
                        imgs.append(img)
                        cam_frames[cam_name] = img

                    for arm in ARM_TOPICS:
                        arm_name = arm.name
                        state_orig_ts = next_obs.get(f"orig_qpos_ts_{arm_name}", 0)
                        state_orig_ts_dict[arm_name] = state_orig_ts

                    # create a horizontal concat if there are at least two views
                    if len(imgs) >= 2:
                        frame = np.hstack([np.rot90(img, 2) if evo1_config.rotate_for_save else img for img in imgs[:2]])
                    else:
                        frame = imgs[0] if imgs else np.zeros((224, 224, 3), dtype=np.uint8)

                    cam_frames[cam_name] = img
                    if self.write_logs and frames is not None:
                        frames.append(frame.astype(np.uint8))

                    # 记录 CSV + 保存图片
                    if self.write_logs and self.episode_logger:
                        self.episode_logger.record_step(
                            ep=ep_idx,
                            step=steps_taken,
                            h=h,
                            obs=next_obs,
                            action=publish_action,
                            frame_images=cam_frames,
                            orig_image_ts=cam_orig_ts_dict,
                            orig_state_ts=state_orig_ts_dict,
                        )
                except Exception:
                    logger.exception("Frame processing failed; skipping frame.")

                steps_taken += 1
                logger.info(f"[ep{ep_idx + 1}] step {step_idx} horizon {h} published, steps_taken={steps_taken}")

                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"Step overran: took {elapsed:.4f}s (> {dt:.4f}s)")

        # end episode
        self.total_episodes += 1

        # save video for episode
        if self.write_logs and frames is not None and self.episode_logger:
            video_name = f"{__class__.__name__}_ep{ep_idx + 1}.mp4"
            self.episode_logger.submit_video(frames, video_name, fps=20)

        logger.info(f"Episode {ep_idx + 1} finished. steps={steps_taken}")
        return {"success": True, "steps": steps_taken}

    async def run(self, num_episodes: int = evo1_config.num_episodes):
        logger.info(f"Connecting to policy server: {self.server_url}")
        async with websockets.connect(self.server_url, ping_interval=10, ping_timeout=60) as ws:
            logger.info("Connected to server.")

            for ep in range(num_episodes):
                res = await self.run_episode(ws, ep_idx=ep, max_steps=evo1_config.max_steps)
                logger.info(f"Episode result: {res}")

        logger.info("All episodes finished.")

        if self.write_logs and self.episode_logger:
            self.episode_logger.close()


def main():
    global ARM_TOPICS, CAMERA_TOPICS

    config = get_config()

    datacenter = InteractionDataCenter(config)
    ARM_TOPICS = config.get_ava_arms()
    CAMERA_TOPICS = config.get_ava_cameras()
    datacenter.start()

    evaluator = Evo1RealRobotEvaluator(datacenter, server_url=evo1_config.server_url)
    try:
        asyncio.run(evaluator.run(num_episodes=evo1_config.num_episodes))
    finally:
        if evaluator.write_logs and evaluator.episode_logger:
            evaluator.episode_logger.close()
        datacenter.stop()


if __name__ == "__main__":
    main()
