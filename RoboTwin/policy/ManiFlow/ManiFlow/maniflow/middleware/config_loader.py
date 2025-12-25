import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("config_loader")


# ======================== 配置类定义 ========================
@dataclass
class ArmConfig:
    """单个机械臂配置"""

    name: str
    enabled: bool = True
    base_topic: str = ""
    dof: int = 6  # 关节自由度（不含夹爪）


@dataclass
class CameraConfig:
    """单个相机配置"""

    name: str
    enabled: bool = True
    base_topic: str = ""
    rgb_enabled: bool = True
    depth_enabled: bool = False


@dataclass
class Evo1Config:
    """Evo-1 评估配置"""

    server_url: str = "ws://localhost:8000"
    num_episodes: int = 10
    horizon: int = 1
    max_steps: int = 1000
    log_save_dir: str = "./eval_logs"
    seed: int = 42
    write_logs: bool = True
    action_timeout: float = 10.0
    rotate_for_save: bool = False
    send_freq: int = 20
    image_mask_len: int = 3
    action_mask_len: int = 24
    prompt: str = "do something."

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evo1Config":
        return cls(
            server_url=data.get("server_url", "ws://localhost:8000"),
            num_episodes=data.get("num_episodes", 10),
            horizon=data.get("horizon", 1),
            max_steps=data.get("max_steps", 1000),
            log_save_dir=data.get("log_save_dir", "./eval_logs"),
            seed=data.get("seed", 42),
            write_logs=data.get("write_logs", True),
            action_timeout=data.get("action_timeout", 10.0),
            rotate_for_save=data.get("rotate_for_save", False),
            send_freq=data.get("send_freq", 20),
            image_mask_len=data.get("image_mask_len", 3),
            action_mask_len=data.get("action_mask_len", 24),
            prompt=data.get("prompt", "do something."),
        )


@dataclass
class RobotTopicConfig:
    """机器人完整配置（支持YAML加载）"""

    # 机械臂配置列表
    arms: List[ArmConfig] = field(default_factory=list)
    # 相机配置列表
    cameras: List[CameraConfig] = field(default_factory=list)
    # 缓存大小配置
    action_buffer_size: int = 200
    state_buffer_size: int = 30
    camera_buffer_size: int = 30
    # 数据同步配置
    block_timeout: float = 100.0
    check_interval: float = 0.01
    timestamp_tolerance: float = 0.03
    sync_target: str = "image"  # 同步目标，可选 "qpos" 或 "image"
    # 默认提示词
    prompt: str = "do something."

    def get_ava_arms(self) -> List[ArmConfig]:
        """获取有序的启用的机械臂配置"""
        return [arm for arm in self.arms if arm.enabled]

    def get_ava_cameras(self) -> List[CameraConfig]:
        """获取有序的启用的相机配置"""
        return [cam for cam in self.cameras if cam.enabled]

    def get_arm_dof(self, arm_name: str) -> int:
        """获取指定机械臂的DOF"""
        for arm in self.arms:
            if arm.name == arm_name and arm.enabled:
                return arm.dof
        return 6

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "RobotTopicConfig":
        """从YAML配置文件加载配置"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotTopicConfig":
        """从字典加载配置"""
        # 解析机械臂配置
        arms = []
        arms_data = data.get("arms", {})
        for arm_name, arm_cfg in arms_data.items():
            if isinstance(arm_cfg, dict):
                arms.append(
                    ArmConfig(
                        name=arm_name,
                        enabled=arm_cfg.get("enabled", True),
                        base_topic=arm_cfg.get("base_topic", ""),
                        dof=arm_cfg.get("dof", 6),
                    )
                )

        # 解析相机配置
        cameras = []
        cameras_data = data.get("cameras", {})
        for cam_name, cam_cfg in cameras_data.items():
            if isinstance(cam_cfg, dict):
                streams = cam_cfg.get("streams", {})
                cameras.append(
                    CameraConfig(
                        name=cam_name,
                        enabled=cam_cfg.get("enabled", True),
                        base_topic=cam_cfg.get("base_topic", ""),
                        rgb_enabled=streams.get("rgb", True),
                        depth_enabled=streams.get("depth", False),
                    )
                )

        # 解析缓存配置
        buffers: dict = data.get("buffers", {})
        action_buffer_size = buffers.get("action_buffer_size", 200)
        state_buffer_size = buffers.get("state_buffer_size", 30)
        camera_buffer_size = buffers.get("camera_buffer_size", 30)

        # 解析同步配置
        sync: dict = data.get("sync", {})
        block_timeout = sync.get("block_timeout", 100.0)
        check_interval = sync.get("check_interval", 0.01)
        timestamp_tolerance = sync.get("timestamp_tolerance", 0.03)
        sync_target = sync.get("sync_target", "image")

        # 默认提示词
        prompt = data.get("prompt", "do something.")

        return cls(
            arms=arms,
            cameras=cameras,
            action_buffer_size=action_buffer_size,
            state_buffer_size=state_buffer_size,
            camera_buffer_size=camera_buffer_size,
            block_timeout=block_timeout,
            check_interval=check_interval,
            timestamp_tolerance=timestamp_tolerance,
            prompt=prompt,
            sync_target=sync_target,
        )

    def validate(self) -> List[str]:
        """验证配置有效性，返回错误列表"""
        errors = []
        enabled_arms = [arm for arm in self.arms if arm.enabled]
        enabled_cams = [cam for cam in self.cameras if cam.enabled]

        if not enabled_arms:
            errors.append("至少需要启用一个机械臂")
        if not enabled_cams:
            errors.append("至少需要启用一个相机")

        for arm in enabled_arms:
            if not arm.base_topic:
                errors.append(f"机械臂 '{arm.name}' 缺少 base_topic")
            if arm.dof <= 0:
                errors.append(f"机械臂 '{arm.name}' DOF 必须大于 0")

        for cam in enabled_cams:
            if not cam.base_topic:
                errors.append(f"相机 '{cam.name}' 缺少 base_topic")

        return errors


# ======================== 全局配置管理 ========================
_global_config: RobotTopicConfig | None = None
_global_evo1_config: Evo1Config | None = None


def load_config(yaml_path: Union[str, Path, dict]) -> RobotTopicConfig:
    """加载配置文件并设置为全局配置（同时加载 Evo1Config）"""
    global _global_config, _global_evo1_config

    if isinstance(yaml_path, dict):
        data = yaml_path
        _global_config = RobotTopicConfig.from_dict(data)
    else:
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _global_config = RobotTopicConfig.from_dict(data)

    # 同时加载 Evo-1 配置
    evo1_data = data.get("Evo-1", {})
    prompt = _global_config.prompt
    evo1_data["prompt"] = prompt  # 使用全局配置中的提示词
    _global_evo1_config = Evo1Config.from_dict(evo1_data)

    return _global_config


def get_config() -> RobotTopicConfig:
    """获取全局配置（必须先调用 load_config）"""
    if _global_config is None:
        logger.warning("配置未加载，正在加载默认配置 config.yaml")
        load_config("config.yaml")
    return _global_config


def get_evo1_config() -> Evo1Config:
    """获取 Evo-1 评估配置（必须先调用 load_config）"""
    if _global_evo1_config is None:
        logger.warning("配置未加载，正在加载默认配置 config.yaml")
        load_config("config.yaml")
    return _global_evo1_config


def set_config(config: RobotTopicConfig) -> None:
    """手动设置全局配置"""
    global _global_config
    _global_config = config


def set_evo1_config(config: Evo1Config) -> None:
    """手动设置 Evo-1 配置"""
    global _global_evo1_config
    _global_evo1_config = config


def is_config_loaded() -> bool:
    """检查配置是否已加载"""
    return _global_config is not None


if __name__ == "__main__":
    # 测试配置加载
    config = get_config()
    logger.info(f"加载的配置: {config}")
    logger.info(f"Evo-1 配置: {get_evo1_config()}")
    logger.info(f"启用的相机: {config.get_ava_cameras()}")
