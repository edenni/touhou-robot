import os
import random
import math
from itertools import count

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Independent, Normal
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import BasicLogger
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.policy import DQNPolicy, PPOPolicy
from tianshou.data import ReplayBuffer, Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Recurrent
from tianshou.utils.net.discrete import Actor, Critic

import config as cfg
from env import THEnv
from model import Recurrent


device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = Recurrent(layer_num=2, 
    state_shape=cfg.state_shape, 
    action_shape=cfg.action_shape, 
    device='cuda').cuda()

actor = Actor(preprocess_net=net, 
    action_shape=cfg.action_shape, 
    hidden_sizes=[128, 128], 
    device='cuda').cuda()

critic = Critic(preprocess_net=net, 
    hidden_sizes=[128, 128], 
    device='cuda').cuda()

for m in list(actor.modules()) + list(critic.modules()):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.zeros_(m.bias)

optim = torch.optim.Adam(set(
        actor.parameters()).union(critic.parameters()), lr=0.0003)

dist = torch.distributions.Categorical
policy = PPOPolicy(actor, critic, optim, dist,
    max_grad_norm=cfg.max_grad_norm,
    discount_factor=0.8,
    reward_normalization=False)

env = THEnv(cfg.roi)
env = DummyVectorEnv([lambda: env])

buffer = ReplayBuffer(cfg.capacity, stack_num=cfg.n_stack)
train_collector = Collector(
    policy=policy, 
    env=env,
    buffer=buffer,
    exploration_noise=True)
test_collector = Collector(policy, env, exploration_noise=True)

# log
log_path = os.path.join('logs', 'ppo')
writer = SummaryWriter(log_path)
logger = BasicLogger(writer)

result = onpolicy_trainer(policy, train_collector, test_collector, 
    max_epoch=100, 
    step_per_epoch=10000, 
    repeat_per_collect=2, 
    episode_per_test=3, 
    batch_size=cfg.batch_size, 
    episode_per_collect=6,
    logger=logger)

print(f'Finished training! Use {result["duration"]}')