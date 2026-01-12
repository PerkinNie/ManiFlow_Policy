import sys
import os
import pathlib
import logging
import socket
import numpy as np
from typing import Optional, Dict, Any

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ManiFlow'))
sys.path.append(os.path.join(ROOT_DIR, 'ManiFlow', 'maniflow'))

from inference_framework.config import InferenceConfig, get_config
from inference_framework.utils import (
    create_client_socket, close_socket,
    send_data, receive_data,
    serialize_observation, deserialize_action
)
from middleware.datacenter import InteractionDataCenter
from middleware.utils import datacenter_obs_to_maniflow
from termcolor import cprint


logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("inference_client")


class ManiFlowInferenceClient:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.client_socket: Optional[socket.socket] = None
        self.datacenter: Optional[InteractionDataCenter] = None
        self.is_running = False
        
        logger.info("初始化ManiFlow推理客户端")
        self._init_datacenter()
    
    def _init_datacenter(self):
        logger.info("初始化数据交互中心...")
        try:
            self.datacenter = InteractionDataCenter(
                config_path=self.config.datacenter.config_path
            )
            logger.info("数据交互中心初始化成功")
        except Exception as e:
            logger.error(f"数据交互中心初始化失败: {e}")
            raise
    
    def connect(self) -> bool:
        logger.info(f"连接到服务端: {self.config.client.server_host}:{self.config.client.server_port}")
        self.client_socket = create_client_socket(
            self.config.client.server_host,
            self.config.client.server_port,
            self.config.client.timeout
        )
        
        if not self.client_socket:
            logger.error("无法连接到服务端")
            return False
        
        logger.info("成功连接到服务端")
        return True
    
    def disconnect(self):
        if self.client_socket:
            close_socket(self.client_socket)
            self.client_socket = None
            logger.info("已断开与服务端的连接")
    
    def _get_observation(self) -> Optional[Dict[str, Any]]:
        try:
            raw_obs = self.datacenter.get_observation()
            if not raw_obs:
                logger.warning("获取观测失败或超时")
                return None
            
            policy_obs = datacenter_obs_to_maniflow(raw_obs)
            logger.info(f"获取观测成功: qpos.shape={raw_obs.get('qpos', np.array([])).shape}")
            return policy_obs
        except Exception as e:
            logger.error(f"获取观测异常: {e}")
            return None
    
    def _request_action(self, observation: Dict[str, Any], task_name: str = "dual_arm_pick_box") -> Optional[np.ndarray]:
        try:
            request_data = serialize_observation(observation, task_name)
            
            if not send_data(self.client_socket, request_data):
                logger.error("发送观测数据失败")
                return None
            
            response_data = receive_data(self.client_socket, timeout=self.config.client.timeout)
            if not response_data:
                logger.error("接收动作数据失败")
                return None
            
            response = deserialize_action(response_data)
            
            if not response.success:
                logger.error(f"服务端推理失败: {response.message}")
                return None
            
            logger.info(f"接收动作成功: shape={response.action.shape}")
            return response.action
        except Exception as e:
            logger.error(f"请求动作异常: {e}")
            return None
    
    def _publish_action(self, action: np.ndarray) -> bool:
        try:
            success = self.datacenter.publish_action(action)
            if success:
                logger.info(f"动作发布成功: {action}")
            else:
                logger.error("动作发布失败")
            return success
        except Exception as e:
            logger.error(f"发布动作异常: {e}")
            return False
    
    def run_inference(self, max_steps: int = 100, task_name: str = "dual_arm_pick_box"):
        logger.info("开始推理...")
        
        if not self.connect():
            logger.error("连接服务端失败")
            return
        
        self.datacenter.start()
        self.is_running = True
        
        try:
            for step_idx in range(max_steps):
                logger.info(f"===== 第 {step_idx + 1} 步 =====")
                input(f"按 Enter 获取第 {step_idx+1} 步的观测")
                raw_obs = self._get_observation()
                if not raw_obs:
                    cprint(f"❌ 第{step_idx+1}步：获取观测失败，终止推理", "red")
                    break
                
                action = self._request_action(raw_obs, task_name)
                if action is None or action.size == 0:
                    cprint(f"❌ 第{step_idx+1}步：获取动作失败，终止推理", "red")
                    break
                
                n_action_steps = self.config.action.n_action_steps
                action_steps = min(n_action_steps, action.shape[0])
                
                for h in range(action_steps):
                    single_action = action[h]
                    # publish_action = single_action[:self.config.observation.arm_dofs]
                    publish_action = single_action
                    input(f"按 Enter 执行第 {h+1} 个动作")
                    logger.info(f"执行动作 {h+1}/{action_steps}: {publish_action}")
                    
                    next_obs = self.datacenter.step(publish_action)
                    if not next_obs:
                        logger.warning("动作执行后未获取到观测")
                        break
                
                logger.info(f"===== 第 {step_idx + 1} 步完成 =====\n")
            
            cprint(f"📝 推理完成 | 总步数: {step_idx + 1}", "green")
        
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止推理")
        except Exception as e:
            logger.error(f"推理过程异常: {e}")
            raise
        finally:
            self.stop()
    
    def reset(self):
        logger.info("重置客户端状态...")
        if self.client_socket:
            self.disconnect()
        self.is_running = False
    
    def stop(self):
        logger.info("停止推理客户端...")
        self.is_running = False
        
        if self.datacenter:
            self.datacenter.stop()
        
        self.disconnect()
        logger.info("客户端已停止")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ManiFlow推理客户端")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--host", type=str, default="localhost", help="服务端地址")
    parser.add_argument("--port", type=int, default=5000, help="服务端端口")
    parser.add_argument("--max-steps", type=int, default=100, help="最大推理步数")
    parser.add_argument("--task-name", type=str, default="dual_arm_pick_box", help="任务名称")
    
    args = parser.parse_args()
    
    config = get_config(args.config)
    if args.host:
        config.client.server_host = args.host
    if args.port:
        config.client.server_port = args.port
    
    client = ManiFlowInferenceClient(config)
    
    try:
        client.run_inference(max_steps=args.max_steps, task_name=args.task_name)
    except KeyboardInterrupt:
        logger.info("收到中断信号")
    finally:
        client.stop()


if __name__ == "__main__":
    main()
