if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
import sys

MANIFLOW_ROOT = str(pathlib.Path(__file__).parent.parent.parent.parent)
sys.path.append(MANIFLOW_ROOT)
sys.path.append(os.path.join(MANIFLOW_ROOT, 'ManiFlow'))
sys.path.append(os.path.join(MANIFLOW_ROOT, 'ManiFlow', 'maniflow'))

sys.path.insert(0, '../../../')
sys.path.append('ManiFlow/env_runner')
sys.path.append('ManiFlow/maniflow/policy')
sys.path.append('ManiFlow')
sys.path.append('ManiFlow/maniflow')

from hydra.core.hydra_config import HydraConfig
from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy
from maniflow.dataset.base_dataset import BaseDataset
from maniflow.env_runner.robot_runner import RobotRunner
from maniflow.env_runner.openmind_runner import OpenmindRunner
from maniflow.common.checkpoint_util import TopKCheckpointManager
from maniflow.common.pytorch_util import dict_apply, optimizer_to
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.model.common.lr_scheduler import get_scheduler

from config_loader import ArmConfig, CameraConfig, get_config, get_evo1_config
from datacenter import InteractionDataCenter
from utils import datacenter_obs_to_evo1, save_video

ARM_TOPICS = []
CAMERA_TOPICS = []

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainManiFlowRoboTwinWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, datacenter: InteractionDataCenter, output_dir=None):
        self.cfg = cfg
        self.datacenter = datacenter
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: ManiFlowTransformerImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: ManiFlowTransformerImagePolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

    def eval(self, mode='best'):
        # load the latest checkpoint
        cfg = copy.deepcopy(self.cfg)
        
        lastest_ckpt_path = self.get_checkpoint_path(tag=mode, monitor_key=cfg.checkpoint.topk.monitor_key)
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from {mode} checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
            # print ckpt info
            cprint(f"{self.epoch} epochs, {self.global_step} steps", 'magenta')
        else:
            cprint(f"Checkpoint {lastest_ckpt_path} does not exist!", 'red')
        
        n_obs_steps = cfg['n_obs_steps']
        n_action_steps = cfg['n_action_steps']

        # configure env
        env_runner = OpenmindRunner(
            output_dir=output_dir,
            n_obs_steps=n_obs_steps, 
            n_action_steps=n_action_steps)
        self._output_dir = output_dir # recover output_dir
        
        policy = self.model
        datacenter = self.datacenter
        if cfg.training.use_ema:
            policy = self.ema_model
    
        policy.eval()
        policy.cuda()

        # inference_steps = cfg.policy.num_inference_steps
        all_rollout_steps = [10] # [10, 1, 4, 2, 8]
        for inference_steps in all_rollout_steps:
            eval_episodes = cfg.robotwin_task.env_runner.eval_episodes
            cprint(f"Running evaluation for {inference_steps} inference steps", 'magenta')

            horizon = policy.horizon
            n_action_steps = policy.n_action_steps
            cprint(f"Evaluating with horizon={horizon}, n_action_steps={n_action_steps}, eval_episodes={eval_episodes}, inference_steps={inference_steps}", 'magenta')

            # Create eval results directory
            eval_dir = os.path.join(self.output_dir, f'eval_results/{self.epoch}/eval_{eval_episodes}_episodes/horizon{horizon}_act{n_action_steps}/{inference_steps}')
            os.makedirs(eval_dir, exist_ok=True)

            policy.num_inference_steps = inference_steps
            runner_log = env_runner.run(policy, datacenter)

            # cprint(f"---------------- Eval Results --------------", 'magenta')
            # metrics_dict = {}
            # for key, value in runner_log.items():
            #     if isinstance(value, float):
            #         metrics_dict[key] = value
            #         cprint(f"{key}: {value:.4f}", 'magenta')
            #     if isinstance(value, dict):
            #         for k, v in value.items():
            #             if isinstance(v, float):
            #                 metrics_dict[f"{key}/{k}"] = v
            #                 cprint(f"{key}/{k}: {v:.4f}", 'magenta')
            
            # # Save metrics to JSON
            # import json
            # metrics_path = os.path.join(eval_dir, f'metrics_{mode}_{self.epoch}.json')
            # with open(metrics_path, 'w') as f:
            #     json.dump(metrics_dict, f, indent=4)
            
            # # Save videos if they exist in runner_log
            # runner_log.pop('average_success_rate', None) # Remove average_success_rate from runner_log
            # video_id = 0
            # task_name = runner_log['task_name']
            # for k, v in runner_log.items():
            #     if 'video' in k:
            #         if isinstance(v, np.ndarray):
            #             video_dir = os.path.join(eval_dir, 'videos', task_name)
            #             os.makedirs(video_dir, exist_ok=True)
            #             video_path = os.path.join(video_dir, f'{k}_{mode}_{self.epoch}_{video_id}.mp4')
                        
            #             # Convert from N, C, H, W to N, H, W, C format for saving
            #             v = np.transpose(v, (0, 2, 3, 1))
            #             # Save video using imageio or cv2
            #             import imageio
            #             imageio.mimsave(video_path, v, fps=10)
            #         elif hasattr(v, '_path'):  # Handle wandb.Video object
            #             video_dir = os.path.join(eval_dir, 'videos', task_name)
            #             os.makedirs(video_dir, exist_ok=True)
            #             video_path = os.path.join(video_dir, f'{k}_{mode}_{self.epoch}_{video_id}.mp4')
            #             # Copy the video file from wandb path to our eval directory
            #             shutil.copy2(v._path, video_path)
            #         else:
            #             cprint(f"Unknown video format for {k}", 'red')
            #         video_id += 1
            # cprint(f"Evaluation results saved to {eval_dir}", 'magenta')

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def get_checkpoint_path(self, tag='latest', monitor_key='test_mean_score'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10 if 'loss' not in monitor_key else float('inf')
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                try:
                    # Extract score for the specified monitor_key
                    score_str = ckpt.split(f'{monitor_key}=')[1].split('.ckpt')[0]
                    score = float(score_str)
                    
                    # Update best score based on whether we're minimizing or maximizing
                    if 'loss' in monitor_key:
                        if score < best_score:
                            best_ckpt = ckpt
                            best_score = score
                    else:
                        if score > best_score:
                            best_ckpt = ckpt
                            best_score = score
                except (IndexError, ValueError):
                    # Skip checkpoints that don't have the monitor_key
                    continue
            
            if best_ckpt is None:
                raise ValueError(f"No checkpoints found with monitor key: {monitor_key}")
            
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            
    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.parent.joinpath('config'))
)
def main(cfg):
    global ARM_TOPICS, CAMERA_TOPICS

    config = get_config()

    datacenter = InteractionDataCenter(config)
    ARM_TOPICS = config.get_ava_arms()
    CAMERA_TOPICS = config.get_ava_cameras()
    datacenter.start()

    workspace = TrainManiFlowRoboTwinWorkspace(cfg, datacenter)
    workspace.eval()

    # datacenter.stop()

if __name__ == "__main__":
    main()
