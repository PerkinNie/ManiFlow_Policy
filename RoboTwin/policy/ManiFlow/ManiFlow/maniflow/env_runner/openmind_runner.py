import wandb
import numpy as np
import torch
import tqdm
import cv2
import json
import logging

from maniflow.policy.base_policy import BasePolicy
from maniflow.common.pytorch_util import dict_apply
from maniflow.env_runner.base_runner import BaseRunner
import maniflow.common.logger_util as logger_util
from queue import deque
from termcolor import cprint
from maniflow.middleware.utils import datacenter_obs_to_maniflow
from maniflow.middleware.datacenter import InteractionDataCenter

def init_logger():
    logger = logging.getLogger("ImgPubNode")  # 创建logger实例
    logger.setLevel(logging.INFO)  # 设置日志级别
    # 配置控制台输出格式
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger

logger = init_logger()

class OpenmindRunner(BaseRunner):
    def __init__(self,
                 output_dir=None,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=1,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        cprint(f"OpenmindRunner for task {self.task_name}", 'green')

        steps_per_render = max(10 // fps, 1)

        self.eval_episodes = eval_episodes
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        self.obs = deque(maxlen=n_obs_steps+1)

        cprint(f"OpenmindRunner initialized with n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}", "green")
        self.env = None


    def stack_last_n_obs(self, all_obs, n_steps):
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
    def get_action(self, policy: BasePolicy, observaton=None) -> bool: # by tianxing chen
        device, dtype = policy.device, policy.dtype
        if observaton is not None:
            self.obs.append(observaton)  # update
        obs = self.get_n_steps_obs()
        
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
            obs_dict_input['task_name'] = [self.task_name]
            action_dict = policy.predict_action(obs_dict_input)
            
        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
        action = np_action_dict['action'].squeeze(0)
        return action

    def run(self, policy: BasePolicy, datacenter: InteractionDataCenter, ep_idx: int, max_steps: int):
        logger.info(f"正在推理第 {ep_idx + 1} 个Epoch...")
        try:
            for step_idx in range(max_steps):
                input("等待用户确认请求推理")
                raw_obs = datacenter.get_observation()
                if not raw_obs:
                    cprint(f"❌ 第{ep_idx+1}轮第{step_idx}步：获取观测失败，终止本轮", "red")
                    break

                try:
                    policy_obs = datacenter_obs_to_maniflow(raw_obs)
                except Exception as e:
                    cprint(f"❌ 第{ep_idx+1}轮第{step_idx}步：观测格式转换失败 - {str(e)}", "red")
                    break

                self.update_obs(policy_obs)

                # 4. 模型推理动作
                try:
                    action = self.get_action(policy)
                except Exception as e:
                    cprint(f"❌ 第{ep_idx+1}轮第{step_idx}步：模型推理失败 - {str(e)}", "red")
                    break

                # 5. 发布动作到机器人
                for h in range(min(self.n_action_steps, action.shape[0])):
                    single_action = action[h]
                    # if single_action.size < self.arm_dofs:
                    #     # pad with zeros to match expectation
                    #     pad_len = self.arm_dofs - single_action.size
                    #     single_action = np.concatenate([single_action, np.zeros(pad_len, dtype=float)])

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
                    logger.info(f"发布动作 第{h}步：{publish_action} 总共{action.shape[0]} 步")
                    input("等待用户确认再次发布动作")
                    next_obs = self.datacenter.step(publish_action)
                    if not next_obs:
                        logger.warning("No observation after action publish; stopping.")
                        break
                
            # cprint(f"📝 第{episode_idx+1}轮结束 | 总步数: {step_idx} | 任务完成: {episode_success}", "green")

        except Exception as e:
            cprint(f"\n❌ 评估过程异常终止 - {str(e)}", "red")
            raise
        finally:
            datacenter.stop()
            cprint("\n🛑 交互中心已停止", "green")


if __name__ == '__main__':
    test = OpenmindRunner('./')
    print('ready')