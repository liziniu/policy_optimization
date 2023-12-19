import numpy as np
import copy
from typing import List
from envs.linear_bandit import LinearBandit
from utils.collect_data import Transition, ret_uniform_policy
from utils.utils import softmax


class PolicyGradient:
    def __init__(
        self,
        feature_func,
        reward_func,
        ref_policy,
        feature_dim: int,
        action_num: int,
        reg_coef: float,
        step_size: float,
        num_iters: int,
        is_adaptive: bool,
        ada_coef: float,
        eval_times: int = 5,
        logger=None,
    ) -> None:
        self.feature_func = feature_func
        self.reward_func = reward_func
        self.ref_policy = ref_policy

        self.feature_dim = feature_dim
        self.action_num = action_num
        self.reg_coef = reg_coef
        self.step_size = step_size
        self.num_iters = num_iters

        self.is_adaptive = is_adaptive
        self.ada_coef = ada_coef

        self.hist_grad_squared_norm = 0.0
        self.eval_times = eval_times
        self.logger = logger

        # initialize the policy parameters
        self.policy_param = np.random.uniform(0, 1, self.feature_dim)

    def ret_action_prob(self, state: np.ndarray) -> np.ndarray:
        """
        Calculate the action probability at the input state.
        """
        action_prob = np.array(
            [
                np.dot(self.policy_param, self.feature_func(state, action_idx))
                for action_idx in range(self.action_num)
            ],
            dtype=np.float32,
        )
        action_prob = np.exp(action_prob)
        action_prob = action_prob / np.sum(action_prob)
        return action_prob

    @property
    def get_param(self) -> np.ndarray:
        return self.policy_param

    def ret_policy(self):
        action_num = self.action_num
        feature_func = copy.deepcopy(self.feature_func)
        param = self.policy_param

        def policy(state: np.ndarray) -> np.ndarray:
            arr = np.zeros(action_num, np.float32)
            for action_idx in range(action_num):
                feature = feature_func(state, action_idx)
                arr[action_idx] = np.dot(feature, param)
            prob = softmax(arr)

            return prob

        return policy

    def update_once(self, dataset: List[np.ndarray]) -> (float, float):
        grad = np.zeros_like(self.policy_param, np.float32)
        for state in dataset:
            rews = [
                self.reward_func(state, action_idx)
                for action_idx in range(self.action_num)
            ]
            rews = np.array(rews, np.float32)
            cur_policy_action_prob = self.ret_action_prob(state)
            ref_policy_action_prob = self.ref_policy(state)
            log_ratio = np.log(cur_policy_action_prob) - np.log(ref_policy_action_prob)

            # (|A|) vector
            full_vec = cur_policy_action_prob * (rews - self.reg_coef * log_ratio)
            # (|A|*dim) matrix
            score_mat = self.cal_score_mat(state)
            neg_cur_data_grad = np.matmul(score_mat.T, full_vec)
            # single_grad = np.matmul(single_grad, full_vec)

            grad -= neg_cur_data_grad

        grad = grad / len(dataset)
        self.hist_grad_squared_norm += np.sum(np.square(grad))
        if self.is_adaptive:
            step_size = self.ada_coef / np.sqrt(self.hist_grad_squared_norm)
        else:
            step_size = self.step_size

        self.policy_param -= step_size * grad

        return step_size, np.sqrt(np.sum(np.square(grad)))

    def cal_score_mat(self, state: np.ndarray):
        """
        calculate score function matrix delta_{theta} log (pi_theta (|s)) with the size of |A| * feature_dim
        """
        score_list = []
        feature_mat = np.stack(
            [
                self.feature_func(state, action_idx)
                for action_idx in range(self.action_num)
            ],
            axis=0,
        )
        act_prob = self.ret_action_prob(state)
        avg_feature = np.matmul(feature_mat.T, act_prob)
        assert avg_feature.shape == (
            self.feature_dim,
        ), "The average feature is invalid."
        for action_idx in range(self.action_num):
            score_list.append(self.feature_func(state, action_idx) - avg_feature)

        score_mat = np.stack(score_list, axis=0)
        # print()
        assert score_mat.shape == (self.action_num, self.feature_dim)
        return score_mat

    def evaluate_loss(self, dataset: List[np.ndarray]) -> float:
        loss = 0.0
        for state in dataset:
            rews = [
                self.reward_func(state, action_idx)
                for action_idx in range(self.action_num)
            ]
            rews = np.array(rews, np.float32)
            cur_act_prob = self.ret_action_prob(state)
            avg_rew = np.dot(rews, cur_act_prob)

            ref_act_prob = self.ref_policy(state)
            log_ratio = np.log(cur_act_prob) - np.log(ref_act_prob)
            kl = np.dot(cur_act_prob, log_ratio)

            aug_rew = avg_rew - self.reg_coef * kl
            loss -= aug_rew
        loss /= len(dataset)

        return loss

    def evaluate_reward(self, env) -> float:
        policy = self.ret_policy()
        rew = env.evaluate_reward(policy)

        return rew

    def train(self, dataset: List[np.ndarray], env: LinearBandit, learned_env) -> float:
        for step in range(self.num_iters):
            step_size, grad_norm = self.update_once(dataset)
            eval_interval = max(1, int(self.num_iters / self.eval_times))
            if step % eval_interval == 0:
                loss = self.evaluate_loss(dataset)
                fake_rew = self.evaluate_reward(learned_env)
                rew = self.evaluate_reward(env)
                self.logger.info(
                    f"Iteration: {step: d}, step size: {step_size: .2f}, gradient norm: {grad_norm: .4f}, loss: {loss: .4f}, fake reward: {fake_rew: .4f}, reward: {rew: .4f}."
                )

        rew = self.evaluate_reward(env)
        return rew
