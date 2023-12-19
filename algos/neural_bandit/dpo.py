import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.logger import Logger


class DirectPreferenceOptimizer:
    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        env,
        learned_env,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        beta: float = 1.0,
        logger: Logger = None,
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.env = env
        self.learned_env = learned_env
        self.batch_size = batch_size
        self.beta = beta
        self.logger = logger

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def optimize_one_epoch(
        self,
        states: torch.tensor,
        positive_actions: torch.tensor,
        negative_actions: torch.tensor,
    ):
        total_loss = 0.0
        total_gradient_norm = 0.0
        k = 0
        for i in range(0, len(states), self.batch_size):
            self.optimizer.zero_grad()

            _states = states[i : i + self.batch_size]
            distributions = self.policy(_states)
            ref_distributions = self.ref_policy(_states)

            _positive_actions = positive_actions[i : i + self.batch_size]
            _negative_actions = negative_actions[i : i + self.batch_size]

            pi_positive_logprobs = distributions[
                np.arange(len(_states)), _positive_actions
            ]
            pi_negative_logprobs = distributions[
                np.arange(len(_states)), _negative_actions
            ]

            ref_positive_logprobs = ref_distributions[
                np.arange(len(_states)), _positive_actions
            ]
            ref_negative_logprobs = ref_distributions[
                np.arange(len(_states)), _negative_actions
            ]

            pi_log_ratios = pi_positive_logprobs - pi_negative_logprobs
            ref_log_ratios = ref_positive_logprobs - ref_negative_logprobs

            log_ratios = pi_log_ratios - ref_log_ratios

            loss = -F.logsigmoid(self.beta * log_ratios).mean()

            total_loss += loss.item()

            loss.backward()

            gradient_norm = 0.0
            for p in self.policy.parameters():
                param_norm = p.grad.detach().data.norm(2)
                gradient_norm += param_norm.item() ** 2
            gradient_norm = gradient_norm**0.5
            total_gradient_norm += gradient_norm

            self.optimizer.step()

            k += 1

        return total_loss / k, total_gradient_norm / k

    def optimize(
        self,
        states: torch.tensor,
        positive_actions: torch.tensor = None,
        negative_actions: torch.tensor = None,
        num_epochs: int = 10,
        optimal_rew: float = 0.0,
    ):
        eval_epoch_interval = 5
        for epoch in range(num_epochs):
            if epoch % eval_epoch_interval == 0:
                true_reward = self.env.evaluate_policy(self.policy).item()
                fake_reward = self.learned_env.evaluate_policy(self.policy).item()
                gap = np.abs(optimal_rew - true_reward)

            loss, gradient_norm = self.optimize_one_epoch(
                states, positive_actions, negative_actions
            )
            if epoch % eval_epoch_interval == 0:
                if self.logger:
                    self.logger.info(
                        f"[Policy] Epoch: {epoch} loss: {loss:.4f} grad norm: {gradient_norm:.4f} "
                        f"true reward: {true_reward:.4f} fake reward: {fake_reward:.4f} gap: {gap:.4f}"
                    )
