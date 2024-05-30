from __future__ import annotations

import random
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal
from typing import List

from omnisafe.models.actor.actor_builder import ActorBuilder
from omnisafe.models.actor_critic.actor_critic import ActorCritic
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.models.base import Actor, Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig

class ConstraintFlippedActorCritic(ConstraintActorCritic):
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        super().__init__(obs_space, act_space, model_cfgs, epochs)

        # 使用ActorBuilder创建两个Actor实例
        actor_builder = ActorBuilder(
            obs_space, act_space, 
            hidden_sizes=model_cfgs.actor.hidden_sizes,
            activation=model_cfgs.actor.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
        )
        self.actor_a = actor_builder.build_actor(model_cfgs.actor_type)
        self.actor_b = actor_builder.build_actor(model_cfgs.actor_type)

        # 创建一个输出标量w的神经网络
        self.w_network = nn.Sequential(
            nn.Linear(obs_space.shape[0], model_cfgs.critic.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(model_cfgs.critic.hidden_sizes[0], 1),
            nn.Sigmoid()
        )

        # 初始化actor和w_network的优化器和scheduler
        self.actor_optimizer = optim.Adam(
            list(self.actor_a.parameters()) + list(self.actor_b.parameters()) + list(self.w_network.parameters()),
            lr=model_cfgs.actor.lr
        )
        self.actor_scheduler = optim.lr_scheduler.LinearLR(
            self.actor_optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs, verbose=True
        )

    def step(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        w = self.w_network(obs)
        rand_num = random.random()

        if rand_num > w.item():
            action = self.actor_a.predict(obs, deterministic)
            log_prob = self.actor_a.log_prob(action)
        else:
            action = self.actor_b.predict(obs, deterministic)
            log_prob = self.actor_b.log_prob(action)

        value_r = self.reward_critic(obs)
        value_c = self.cost_critic(obs)

        return action, value_r[0], value_c[0], log_prob

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        action_a = self.actor_a.predict(obs, deterministic)
        action_b = self.actor_b.predict(obs, deterministic)
        weight = self.w_network(obs)

        # 使用weight的每个元素与随机值进行比较,生成一个布尔掩码
        mask = torch.rand_like(weight) < weight

        # 使用掩码选择action_a或action_b
        action = torch.where(mask, action_a, action_b)

        return action, action_a, action_b, weight
