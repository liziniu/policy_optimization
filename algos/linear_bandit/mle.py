import cvxpy as cp
import numpy as np
from envs.linear_bandit import LinearBandit
from typing import List
from utils.collect_data import (
    Transition,
    sigmoid,
    collect_preference_data,
    ret_uniform_policy,
)


class MLERewardLearning:
    def __init__(
        self,
        feature_func,
        param_dim: int,
        step_size: float,
        num_iters: int,
        is_adaptive: bool,
        ada_coef: float,
    ) -> None:
        self.feature_func = feature_func
        self.param_dim = param_dim
        self.step_size = step_size
        self.num_iters = num_iters

        self.is_adaptive = is_adaptive
        self.ada_coef = ada_coef

        self.hist_grad_squared_norm = 0.0
        # initialize the reward parameters
        self.reward_param = np.random.standard_normal(self.param_dim)
        self.reward_param /= np.sqrt(np.sum(np.square(self.reward_param)))

    @property
    def get_reward_param(self) -> np.ndarray:
        return self.reward_param

    @property
    def get_reward_func(self):
        learned_rew_param = self.reward_param

        def reward_func(state: np.ndarray, action: int):
            reward = np.dot(self.feature_func(state, action), learned_rew_param)
            return reward

        return reward_func

    def update_once(self, dataset: List[Transition]) -> float:
        grad = np.zeros_like(self.reward_param, np.float32)
        equal_num = 0
        diff = 0.0
        for transition in dataset:
            state = transition.state
            action_one = transition.action_0
            action_two = transition.action_1
            pref = transition.pref
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            if pref_act == non_pref_act:
                equal_num += 1
            feature_pref_act, feature_non_pref_act = (
                self.feature_func(state, pref_act),
                self.feature_func(state, non_pref_act),
            )
            rew_pref_act, rew_non_pref_act = (
                np.dot(feature_pref_act, self.reward_param),
                np.dot(feature_non_pref_act, self.reward_param),
            )
            diff += rew_pref_act - rew_non_pref_act
            grad -= sigmoid(rew_non_pref_act - rew_pref_act) * (
                feature_pref_act - feature_non_pref_act
            )

        grad = grad / len(dataset)
        grad_norm = np.sqrt(np.sum(np.square(grad)))
        self.hist_grad_squared_norm += np.sum(np.square(grad))

        if self.is_adaptive:
            step_size = self.ada_coef / np.sqrt(self.hist_grad_squared_norm)
        else:
            step_size = self.step_size
        # print(f'step_size: {step_size: .1f}, grad l2 norm: {grad_squared_norm: .4f}')

        self.reward_param -= step_size * grad
        return grad_norm

    def train(self, dataset: List[Transition], true_reward_param: np.ndarray):
        for step in range(self.num_iters):
            grad_norm = self.update_once(dataset)

            loss, l2_dist = self.evaluate(dataset, true_reward_param)
            eval_interval = max(1, int(self.num_iters / 10.0))
            if step % eval_interval == 0:
                print(
                    f"Iteration: {step: d}, loss: {loss: .4f}, l2 distance: {l2_dist: .4f}, gradient norm: {grad_norm: .6f}."
                )

    def train_by_cvxpy(self, dataset: List[Transition], true_reward_param: np.ndarray):
        psi = cp.Variable(self.param_dim)
        pref_features, non_pref_features = [], []
        for transition in dataset:
            state, action_one, action_two, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.pref,
            )
            if pref == 1:
                pref_act = action_two
                non_pref_act = action_one
            else:
                pref_act = action_one
                non_pref_act = action_two

            feature_pref_act, feature_non_pref_act = (
                self.feature_func(state, pref_act),
                self.feature_func(state, non_pref_act),
            )
            pref_features.append(feature_pref_act)
            non_pref_features.append(feature_non_pref_act)

        pref_features = np.stack(pref_features, axis=0)
        non_pref_features = np.stack(non_pref_features, axis=0)
        reward_diff = (non_pref_features - pref_features) @ psi
        loss = cp.sum(cp.logistic(reward_diff)) / len(dataset)
        problem = cp.Problem(cp.Minimize(loss))

        problem.solve(solver="ECOS", verbose=False)
        psi_arr = np.array(psi.value)
        # set-up the reward parameter
        self.reward_param = psi_arr
        loss, l2_dist, acc = self.evaluate(dataset, true_reward_param)
        # l2_dist = np.sqrt(np.sum(np.square(true_reward_param - psi_arr)))

        return loss, l2_dist, acc

    def evaluate(
        self, dataset: List[Transition], true_reward_param: np.ndarray
    ) -> (float, float, float):
        # calculate the loss
        loss = 0.0
        acc = 0.0
        for transition in dataset:
            state = transition.state
            action_one = transition.action_0
            action_two = transition.action_1
            pref = transition.pref
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            feature_pref_act, feature_non_pref_act = (
                self.feature_func(state, pref_act),
                self.feature_func(state, non_pref_act),
            )
            rew_pref_act, rew_non_pref_act = (
                np.dot(feature_pref_act, self.reward_param),
                np.dot(feature_non_pref_act, self.reward_param),
            )

            loss -= np.log(sigmoid(rew_pref_act - rew_non_pref_act))
            acc += float(rew_pref_act > rew_non_pref_act)

        loss /= len(dataset)
        acc /= len(dataset)

        # calculate the l2 distance with the optimal parameter
        l2_dist = np.sqrt(np.sum(np.square(true_reward_param - self.reward_param)))

        return loss, l2_dist, acc
