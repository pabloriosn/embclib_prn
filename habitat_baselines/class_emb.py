import numpy as np
import random
import torch

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config


class embclip:
    def __init__(self):

        self.path = "configs/eval/ddppo_objectnav_rgb_clip.yaml"

        config = get_config(self.path, [])

        random.seed(config.TASK_CONFIG.SEED)
        np.random.seed(config.TASK_CONFIG.SEED)
        torch.manual_seed(config.TASK_CONFIG.SEED)

        if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
            torch.set_num_threads(1)

        trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
        assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"

        self.trainer = trainer_init(config)

        # self.trainer.load_model()

    def train(self, image, goal=None):
        action = self.trainer.eval(image, goal)
        return action

