import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    max_clients: int = 1


@dataclass
class ClientConfig:
    server_host: str = "localhost"
    server_port: int = 5000
    timeout: int = 30


@dataclass
class ModelConfig:
    config_name: str = "maniflow_image_timm_policy_robotwin2"
    task_name: str = "dual_arm_pick_box"
    alg_name: str = "maniflow_image_timm_policy_robotwin2"
    addition_info: str = "maniflow_image_timm_policy_robotwin2"
    training_seed: int = 0
    
    checkpoint_path: str = field(default_factory=lambda: str(
        Path(__file__).parent.parent / "data" / "outputs" / 
        "dual_arm_pick_box-maniflow_image_timm_policy_robotwin2-1223_seed0" / 
        "checkpoints" / "latest.ckpt"
    ))
    
    config_path: str = field(default_factory=lambda: str(
        Path(__file__).parent.parent / "ManiFlow" / "maniflow" / "config"
    ))


@dataclass
class ObservationConfig:
    resize_size: int = 224
    arm_dofs: int = 7
    image_mask_len: int = 1
    action_mask_len: int = 8


@dataclass
class ActionConfig:
    n_obs_steps: int = 2
    n_action_steps: int = 16
    horizon: int = 16
    temporal_agg: bool = False
    num_queries: int = 15
    max_timesteps: int = 3000
    query_frequency: int = 1


@dataclass
class DataCenterConfig:
    config_path: Optional[str] = None
    block_timeout: float = 10.0
    check_interval: float = 0.01
    timestamp_tolerance: float = 0.1


@dataclass
class InferenceConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    datacenter: DataCenterConfig = field(default_factory=DataCenterConfig)
    
    device: str = "cuda:0"
    log_level: str = "INFO"


def get_config(config_path: Optional[str] = None) -> InferenceConfig:
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return InferenceConfig(**config_dict)
    return InferenceConfig()
