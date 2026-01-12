import sys
import os
import dill
import pathlib
import logging
import socket
import torch
import numpy as np
from typing import Optional

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
# print("===================")
# print(ROOT_DIR)
# input("Press Enter to continue...")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ManiFlow'))
sys.path.append(os.path.join(ROOT_DIR, 'ManiFlow', 'ManiFlow'))
from maniflow_policy import ManiFlow
sys.path.append(os.path.join(ROOT_DIR, 'ManiFlow', 'ManiFlow', 'maniflow'))

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from inference_framework.config import InferenceConfig, get_config
from inference_framework.utils import (
    create_server_socket, close_socket,
    send_data, receive_data,
    serialize_action, deserialize_observation
)
from maniflow.common.pytorch_util import dict_apply
from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy
from collections import deque

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("inference_server")


class ManiFlowInferenceServer:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model: Optional[ManiFlowTransformerImagePolicy] = None
        # self.model: Optional[ManiFlow] = None
        self.server_socket: Optional[socket.socket] = None
        self.is_running = False
        self.n_obs_steps = 2

        self.obs = deque(maxlen=self.n_obs_steps+1)
        
        logger.info("初始化ManiFlow推理服务端")
        self._load_model()
    
    def _load_model(self):
        logger.info("开始加载模型...")
        try:
            usr_args = {
                'config_name': self.config.model.config_name,
                'task_name': self.config.model.task_name,
                'alg_name': self.config.model.alg_name,
                'addition_info': self.config.model.addition_info,
                'training_seed': self.config.model.training_seed,
                'expert_data_num': 100
            }
            
            logger.info(f"配置参数: {usr_args}")
            
            config_path = self.config.model.config_path
            config_name = f"{usr_args['config_name']}.yaml"
            
            config_path = "../../../policy/ManiFlow/ManiFlow/maniflow/config"   # Hydra要求config_path必须是相对路径
            logger.info(f"Hydra配置路径: {config_path}, 配置文件: {config_name}")
            
            logger.info("正在加载Hydra配置...")
            with initialize(config_path=config_path, version_base='1.2'):
                cfg = compose(config_name=config_name)
            logger.info("Hydra配置加载完成")

            parent_directory = str(pathlib.Path(__file__).parent.parent)
            task_name = usr_args['task_name']
            alg_name = usr_args['alg_name']
            addition_info = usr_args['addition_info']
            seed = usr_args['training_seed']
            exp_name = f"{task_name}-{alg_name}-{addition_info}"
            run_dir = os.path.join(parent_directory, "data", "outputs", exp_name + f"_seed{seed}")
            
            logger.info(f"运行目录: {run_dir}")
       
            
            hydra_runtime_cfg = {
                "job": {
                    "override_dirname": usr_args['task_name']
                },
                "run": {
                    "dir": run_dir
                },
                "sweep": {
                    "dir": run_dir,
                    "subdir": "0"
                }
            }
            
            OmegaConf.set_struct(cfg, False)
            cfg.hydra = hydra_runtime_cfg
            cfg.task_name = usr_args["task_name"]
            cfg.expert_data_num = usr_args["expert_data_num"]
            cfg.raw_task_name = usr_args["task_name"]
            OmegaConf.set_struct(cfg, True)
            logger.info("============================")
            logger.info("配置参数:")
            logger.info(cfg)
            logger.info("正在实例化模型...")
            # print(f"cfg.shape_meta is {cfg.shape_meta}")
            # print(f"==============")
            self.model = instantiate(cfg.policy, shape_meta=cfg.shape_meta)
            logger.info("模型实例化完成")
            
            # 从checkpoint加载模型参数
            checkpoint_path = self.config.model.checkpoint_path
            logger.info(f"从checkpoint加载模型参数: {checkpoint_path}")
            
            # 加载checkpoint
            payload = torch.load(checkpoint_path, map_location="cuda:0", pickle_module=dill)
            
            # 检查是否包含模型状态字典
            if "state_dicts" in payload:
                # 获取模型的状态字典
                model_state_dict = payload["state_dicts"]["model"]
                
                # 加载模型参数
                self.model.load_state_dict(model_state_dict)
                logger.info("模型参数加载完成")
            else:
                logger.warning("警告: checkpoint中未找到模型状态字典，使用随机初始化的模型参数")
            
            # 将模型移动到指定设备
            device = torch.device(self.config.device)
            self.model.to(device)
            
            logger.info(f"模型加载成功: {run_dir}")
            logger.info(f"模型类型: {type(self.model).__name__}")
            logger.info(f"模型已移动到设备: {device}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise
    
    
    def _reset_model(self):
        # 重置模型内部状态
        if hasattr(self.model, 't'):
            self.model.t = 0
        
        # 如果模型有时间聚合功能，重置相关状态
        if hasattr(self.model, 'temporal_agg') and self.model.temporal_agg:
            if hasattr(self.model, 'max_timesteps') and hasattr(self.model, 'num_queries') and hasattr(self.model, 'state_dim'):
                self.model.all_time_actions = torch.zeros([
                    self.model.max_timesteps,
                    self.model.max_timesteps + self.model.num_queries,
                    self.model.state_dim,
                ]).to(self.model.device)
            logger.info("重置时间聚合状态")
        else:
            logger.info("模型无时间聚合功能或未启用")
        
        # 重置环境运行器（如果存在）
        if hasattr(self.model, 'env_runner') and self.model.env_runner:
            self.model.env_runner.reset_obs()
        
        logger.info("模型状态已重置")

    @staticmethod
    def stack_last_n_obs(all_obs, n_steps):
        assert(len(all_obs) > 0)
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, 
                dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps,) + all_obs[-1].shape, 
                dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        # support str
        elif isinstance(all_obs[0], str):
            return all_obs * n_steps
        elif isinstance(all_obs[0], (list, tuple)):
            try:
                last_arr = np.asarray(all_obs[-1])
                result = np.zeros((n_steps,) + last_arr.shape,
                                dtype=last_arr.dtype)
                start_idx = -min(n_steps, len(all_obs))
                stacked = np.stack([np.asarray(x) for x in all_obs[start_idx:]])
                result[start_idx:] = stacked
                if n_steps > len(all_obs):
                    result[:start_idx] = result[start_idx]
            except Exception as e:
                raise RuntimeError(f'Failed to convert list/tuple obs to ndarray: {e}')
        else:
            raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
        return result
    
    def reset_obs(self):
        self.obs.clear()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def get_n_steps_obs(self):
        assert(len(self.obs) > 0), 'no observation is recorded, please update obs first'

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs(
                [obs[key] for obs in self.obs],
                self.n_obs_steps
            )
        return result
    
    @torch.no_grad()
    def get_action(self, observaton=None, task_name: str = "") -> bool: # by tianxing chen
        device = self.model.device
        if observaton is not None:
            self.obs.append(observaton)  # update
        obs = self.get_n_steps_obs()
        # print(f"obs {obs}")
        
        # create obs dict
        np_obs_dict = dict(obs)
        # device transfer
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        # run policy
        with torch.no_grad():
            obs_dict_input = {}  # flush unused keys
            # obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
            obs_dict_input['head_cam'] = obs_dict['head_cam'].unsqueeze(0)
            obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
            obs_dict_input['task_name'] = [task_name]
            print("===== obs_dict_input 内容 =====")
            for key, value in obs_dict_input.items():
                print(f"{key}: {value}")
            action_dict = self.model.predict_action(obs_dict_input)
            
        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
        action = np_action_dict['action'].squeeze(0)
        return action
    
    def _predict_action(self, observation: dict, task_name: str) -> np.ndarray:
        try:
            obs = self._encode_obs(observation)
            logger.info(f"输入观测: {obs}")
            
            # 将numpy数组转换为torch tensor并传输到设备
            device = self.model.device
            dtype = self.model.dtype
            
            # 创建观测字典，按照robot_runner.py中的处理方式
            obs_dict_input = {}
            
            # 处理观测数据
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    # 直接传输到设备，不添加额外的批次维度
                    obs_dict_input[key] = torch.from_numpy(value).to(device=device, dtype=dtype)
                else:
                    # 如果不是numpy数组，直接传输到设备
                    obs_dict_input[key] = value.to(device=device, dtype=dtype) if torch.is_tensor(value) else torch.tensor(value, device=device, dtype=dtype)
            
            # 如果task_name存在，也添加到观测字典中
            if task_name and 'task_name' not in obs_dict_input:
                obs_dict_input['task_name'] = [task_name]
            
            # 使用模型的predict_action方法进行预测
            result = self.model.predict_action(obs_dict_input)
            action = result['action']  # 从结果字典中提取动作
            # 如果动作是批次维度的，取第一个
            if len(action.shape) > 2 and action.shape[0] == 1:
                action = action[0]
            return action.detach().cpu().numpy()  # 转换回numpy数组
        except Exception as e:
            logger.error(f"动作预测失败: {e}")
            raise
    
    def start(self):
        logger.info("启动推理服务端...")
        self.server_socket = create_server_socket(
            self.config.server.host,
            self.config.server.port,
            self.config.server.max_clients
        )
        
        if not self.server_socket:
            logger.error("无法创建服务端socket")
            return
        
        self.is_running = True
        logger.info(f"服务端已启动，监听 {self.config.server.host}:{self.config.server.port}")
        
        try:
            while self.is_running:
                logger.info("等待客户端连接...")
                client_socket, client_address = self.server_socket.accept()
                logger.info(f"客户端已连接: {client_address}")
                
                self._handle_client(client_socket)
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止服务端")
        except Exception as e:
            logger.error(f"服务端运行异常: {e}")
        finally:
            self.stop()
    
    def _handle_client(self, client_socket: socket.socket):
        try:
            while self.is_running:
                data = receive_data(client_socket, timeout=None)
                if not data:
                    logger.info("客户端断开连接")
                    break
                
                request = deserialize_observation(data)
                logger.info(f"收到推理请求: task_name={request.task_name}")
                
                try:
                    print(f"------------------")
                    action = self.get_action(request.observation, request.task_name)
                    # print(f"111111111111111111111111")
                    response_data = serialize_action(action, success=True)
                    send_data(client_socket, response_data)
                    logger.info(f"动作预测成功: shape={action.shape}")
                except Exception as e:
                    logger.error(f"推理失败: {e}")
                    error_response = serialize_action(np.array([]), success=False, message=str(e))
                    send_data(client_socket, error_response)
                
        except Exception as e:
            logger.error(f"处理客户端请求异常: {e}")
        finally:
            close_socket(client_socket)
    
    def reset(self):
        logger.info("重置模型状态...")
        self._reset_model()
    
    def stop(self):
        logger.info("停止推理服务端...")
        self.is_running = False
        
        if self.server_socket:
            close_socket(self.server_socket)
            self.server_socket = None
        
        logger.info("服务端已停止")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ManiFlow推理服务端")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务端监听地址")
    parser.add_argument("--port", type=int, default=5000, help="服务端监听端口")
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备")
    
    args = parser.parse_args()
    
    config = get_config(args.config)
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.device:
        config.device = args.device
    
    server = ManiFlowInferenceServer(config)
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("收到中断信号")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
