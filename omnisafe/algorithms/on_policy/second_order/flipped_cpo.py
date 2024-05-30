from omnisafe.algorithms.on_policy.second_order import CPO
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.models.actor_critic.flipped_actor_critic import ConstraintFlippedActorCritic
import numpy as np
import torch
from omnisafe.algorithms import registry
from omnisafe.utils import distributed
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)
from omnisafe.utils.math import conjugate_gradients
from scipy.optimize import minimize

@registry.register
class FlippedCPO(CPO):
    def _init_model(self) -> None:
        self._actor_critic = ConstraintFlippedActorCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)
    def _loss_pi(
        self, 
        obs: torch.Tensor, 
        act: torch.Tensor, 
        logp: torch.Tensor, 
        adv: torch.Tensor,
    ) -> torch.Tensor:
        # 计算权重 w
        features = self._actor_critic.actor.fc_layers(obs)
        w = self._actor_critic.actor.w_head(features)
        # 从 actor 中获取动作分布
        dist = self._actor_critic.actor(obs)
        
        # 获取dist1和dist2,分布
        logp_a = self._actor_critic.actor.log_prob1(act)
        dist = self._actor_critic.actor(obs)
        logp_b = self._actor_critic.actor.log_prob2(act)
        # 计算组合后的对数概率
        logp_ = torch.log(w * torch.exp(logp_a) + (1 - w) * torch.exp(logp_b))
        
        # 计算重要性采样比率
        ratio = torch.exp(logp_ - logp)
        
        # 计算 actor 的损失
        loss = -(ratio * adv).mean()
        
        # 计算熵
        entropy = 1.0
        
        # 记录统计信息
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio.mean().item(),
                'Loss/Loss_pi': loss.item(),
            },
        )
        
        return loss