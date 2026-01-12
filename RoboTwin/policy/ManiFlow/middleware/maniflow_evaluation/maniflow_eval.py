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
from maniflow.env_runner.openmind_runner import OpenmindRunner
from maniflow.middleware.config_loader import get_config, get_maniflow_config
from maniflow.middleware.datacenter import InteractionDataCenter
from maniflow.middleware.utils import datacenter_obs_to_evo1, save_video

ARM_TOPICS = []
CAMERA_TOPICS = []
maniflow_config = get_maniflow_config()

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
        seed = 42
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

    # def eval(self, num_episodes: int = maniflow_config.num_episodes, max_steps: int = maniflow_config.max_steps, mode='best'):
    def eval(self, num_episodes: int = 5, max_steps: int = 200, mode='best'):
        # load the latest checkpoint
        print("Entered maniflow real machine eval stage.")
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
            output_dir=self.output_dir,
            n_obs_steps=n_obs_steps, 
            n_action_steps=n_action_steps)
        self._output_dir = self.output_dir # recover output_dir
        
        policy = self.model
        datacenter = self.datacenter
        if cfg.training.use_ema:
            policy = self.ema_model
    
        policy.eval()
        policy.cuda()

        for episode in num_episodes:
            cprint(f"Running evaluation for {episode} inference steps", 'magenta')

            horizon = policy.horizon
            n_action_steps = policy.n_action_steps
            cprint(f"Evaluating with horizon={horizon}, n_action_steps={n_action_steps}, inference_steps={episode}", 'magenta')

            # policy.num_inference_steps = inference_steps
            runner_log = env_runner.run(policy, datacenter, episode, max_steps)

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
    config_path=str(pathlib.Path(__file__).parent.parent.parent.joinpath('config')),
    config_name="maniflow_image_timm_policy_robotwin2.yaml"
)

# @hydra.main(
#     version_base=None,
#     config_path="/root/workspace/ManiFlow_Policy/RoboTwin/policy/ManiFlow/ManiFlow/maniflow/config",
# )

def main(cfg):
    print(f"========")
    global ARM_TOPICS, CAMERA_TOPICS

    config = get_config()

    datacenter = InteractionDataCenter(config)
    ARM_TOPICS = config.get_ava_arms()
    CAMERA_TOPICS = config.get_ava_cameras()
    datacenter.start()

    print("Loading Maniflow model...")
    workspace = TrainManiFlowRoboTwinWorkspace(cfg, datacenter)
    print("Loaded maniflow model!")
    # workspace = TrainManiFlowRoboTwinWorkspace(cfg)
    # workspace.eval()
    workspace.eval(maniflow_config.num_episodes, maniflow_config.max_steps)

    # datacenter.stop()

if __name__ == "__main__":
    print(f"顺利进入 maniflow_eval 脚本")
    print(str(pathlib.Path(__file__).parent.parent.parent.joinpath('config')))
    main()
