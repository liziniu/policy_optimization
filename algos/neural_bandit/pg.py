import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import Logger


class PolicyModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_num: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        device: str = "cpu",
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_num = action_num

        self.device = torch.device(device)

        network = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            network.append(nn.Linear(hidden_dim, hidden_dim))
            network.append(nn.ReLU())
        network.append(nn.Linear(hidden_dim, action_num))

        self.network = nn.Sequential(*network)

    def forward(self, state: torch.tensor) -> torch.tensor:
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        logits = self.network(state)
        return torch.softmax(logits, dim=-1)


class UniformPolicyModel(nn.Module):
    def __init__(self, action_num: int, device: str = "cpu"):
        super().__init__()
        self.action_num = action_num
        self.device = torch.device(device)

    def forward(self, state: torch.tensor) -> torch.tensor:
        logits = torch.zeros(
            [len(state), self.action_num], dtype=torch.float32, device=self.device
        )
        return torch.softmax(logits, dim=-1)


class PolicyGradientOptimizer:
    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        env,
        learned_env,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        beta: float = 0.0,
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

            with torch.no_grad():
                rews = self.learned_env.evaluate_states(_states)

                kl_divergence = torch.log(distributions) - torch.log(
                    self.ref_policy(_states)
                )

                rews = rews - self.beta * kl_divergence

            assert (
                distributions.shape == rews.shape
            ), f"{distributions.shape} and {rews.shape}"

            loss = -torch.sum(distributions * rews, dim=-1).mean()
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
