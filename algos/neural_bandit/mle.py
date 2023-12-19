import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import Logger


class RewardModel(nn.Module):
    def __init__(
        self,
        state_dim,
        action_num,
        is_target: bool = True,
        state_feature_extractor: nn.Module = None,
        action_feature_extractor: nn.Module = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "relu",
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.is_target = is_target

        activation = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }[activation]

        if state_feature_extractor is None:
            state_feature_extractor = [
                nn.Linear(state_dim, hidden_dim),
                activation,
            ]
            for _ in range(num_layers - 1):
                state_feature_extractor.append(nn.Linear(hidden_dim, hidden_dim))
                state_feature_extractor.append(activation)
        self.state_feature_extractor = nn.Sequential(*state_feature_extractor)
        if action_feature_extractor is None:
            action_feature_extractor = [
                nn.Linear(action_num, hidden_dim),
                activation,
            ]
            for _ in range(num_layers - 1):
                action_feature_extractor.append(nn.Linear(hidden_dim, hidden_dim))
                action_feature_extractor.append(activation)
        self.action_feature_extractor = nn.Sequential(*action_feature_extractor)

        self.predict_layer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, state: torch.tensor, action: torch.tensor) -> torch.tensor:
        assert len(state.shape) == len(action.shape)
        assert torch.all(action >= 0) and torch.all(action <= 1), f"{action}"

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)

        hs = self.state_feature_extractor(state)
        ha = self.action_feature_extractor(action)

        h = torch.cat([hs, ha], dim=1)
        rew = self.predict_layer(h)
        if self.is_target:
            rew = rew + torch.sign(rew)
        return rew


class MaximumLikelihoodEstimator:
    def __init__(
        self,
        action_num: int,
        reward_model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        logger: Logger = None,
    ):
        self.action_num = action_num
        self.reward_model = reward_model
        self.batch_size = batch_size
        self.logger = logger

        self.optimizer = torch.optim.AdamW(
            self.reward_model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def optimize_one_epoch(self, states, positive_actions, negative_actions):
        total_loss = 0.0
        total_acc = 0.0

        k = 0
        for i in range(0, len(states), self.batch_size):
            self.optimizer.zero_grad()

            _states = states[i : i + self.batch_size]
            _positive_actions = positive_actions[i : i + self.batch_size]
            _negative_actions = negative_actions[i : i + self.batch_size]

            _positive_actions = F.one_hot(
                _positive_actions, num_classes=self.action_num
            )
            _negative_actions = F.one_hot(
                _negative_actions, num_classes=self.action_num
            )

            positive_rews = self.reward_model(_states, _positive_actions)
            negative_rews = self.reward_model(_states, _negative_actions)

            loss = -torch.log(torch.sigmoid(positive_rews - negative_rews)).mean()
            loss.backward()
            self.optimizer.step()

            acc = (positive_rews > negative_rews).float().mean()

            total_loss += loss.item()
            total_acc += acc.item()
            k += 1

        return total_loss / k, total_acc / k

    def optimize(self, states, positive_actions, negative_actions, num_epochs):
        for epoch in range(num_epochs):
            loss, acc = self.optimize_one_epoch(
                states, positive_actions, negative_actions
            )
            if self.logger:
                if epoch % 2 == 0:
                    self.logger.info(
                        f"[Reward] Epoch {epoch} loss: {loss:.4f} acc: {acc:.2f}"
                    )
