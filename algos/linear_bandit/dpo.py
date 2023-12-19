import copy
import cvxpy as cp
import numpy as np
from typing import List
from envs.linear_bandit import LinearBandit
from utils.collect_data import Transition, ret_uniform_policy, collect_preference_data
from utils.utils import softmax, sigmoid
from utils.logger import Logger


class DirectPolicyOptimization:
    def __init__(
        self,
        state_dim: int,
        action_num: int,
        feature_dim: int,
        feature_func,
        ref_policy,
        reg_coef: float,
        step_size: float,
        num_iters: int,
        is_adaptive: bool = False,
        ada_coef: float = None,
        logger: Logger = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_num = action_num
        self.feature_dim = feature_dim
        self.feature_func = feature_func
        self.step_size = step_size
        self.num_iters = num_iters
        self.ref_policy = ref_policy
        self.reg_coef = reg_coef
        self.logger = logger

        self.is_adaptive = is_adaptive
        self.ada_coef = ada_coef
        self.hist_grad_squared_norm = 0.0
        # initialize the policy parameter
        self.param = np.random.uniform(0, 1, self.feature_dim)

    def ret_action_prob(self, state: np.ndarray) -> np.ndarray:
        arr = np.zeros(self.action_num, np.float32)
        for action_idx in range(self.action_num):
            feature = self.feature_func(state, action_idx)
            arr[action_idx] = np.dot(feature, self.param)
        prob = softmax(arr)
        return prob

    def ret_policy(self):
        action_num = self.action_num
        feature_func = copy.deepcopy(self.feature_func)
        param = self.param

        def policy(state: np.ndarray) -> np.ndarray:
            arr = np.zeros(action_num, np.float32)
            for action_idx in range(action_num):
                feature = feature_func(state, action_idx)
                arr[action_idx] = np.dot(feature, param)
            prob = softmax(arr)

            return prob

        return policy

    def sample_action(self, state: np.ndarray) -> int:
        prob = self.action_prob(state)
        sampled_act = np.random.choice(a=self.action_num, size=1, replace=True, p=prob)
        return sampled_act

    def update_once(self, dataset: List[Transition]) -> float:
        grad = np.zeros_like(self.param)
        for transition in dataset:
            state, action_one, action_two, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.pref,
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            feat_pref_act, feat_non_pref_act = (
                self.feature_func(state, pref_act),
                self.feature_func(state, non_pref_act),
            )
            cur_policy_act_prob = self.ret_action_prob(state)
            ref_policy_act_prob = self.ref_policy(state)

            log_ratio_diff = self.reg_coef * (
                np.log(cur_policy_act_prob[pref_act] + 1e-6)
                - np.log(ref_policy_act_prob[pref_act] + 1e-6)
                - np.log(cur_policy_act_prob[non_pref_act] + 1e-6)
                + np.log(ref_policy_act_prob[non_pref_act] + 1e-6)
            )
            coef = sigmoid(-log_ratio_diff)
            neg_cur_data_grad = (
                self.reg_coef * coef * (feat_pref_act - feat_non_pref_act)
            )
            grad -= neg_cur_data_grad

        grad /= len(dataset)
        self.hist_grad_squared_norm += np.sum(np.square(grad))
        if self.is_adaptive:
            step_size = self.ada_coef / np.sqrt(self.hist_grad_squared_norm)
        else:
            step_size = self.step_size
        self.param = self.param - step_size * grad
        return np.sqrt(np.sum(np.square(grad)))

    def evaluate_loss(self, dataset: List[Transition], policy=None) -> float:
        """
        Evaluate the loss on the dataset for any policy.
        """
        if policy is None:
            policy = self.ret_policy()

        loss = 0.0
        for transition in dataset:
            state, action_one, action_two, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.pref,
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            eval_policy_act_prob = policy(state)
            ref_policy_act_prob = self.ref_policy(state)
            # if np.isclose(eval_policy_act_prob[pref_act], 0.) or np.isclose(eval_policy_act_prob[non_pref_act], 0.):
            #     print(eval_policy_act_prob[pref_act], eval_policy_act_prob[non_pref_act])
            log_ratio_diff = self.reg_coef * (
                np.log(eval_policy_act_prob[pref_act] + 1e-6)
                - np.log(ref_policy_act_prob[pref_act] + 1e-6)
                - np.log(eval_policy_act_prob[non_pref_act] + 1e-6)
                + np.log(ref_policy_act_prob[non_pref_act] + 1e-6)
            )

            loss -= np.log(sigmoid(log_ratio_diff))
        loss /= len(dataset)
        return loss

    def train(self, dataset: List[Transition], env: LinearBandit) -> float:
        for step in range(self.num_iters):
            grad_norm = self.update_once(dataset)
            if step % 20 == 0:
                loss = self.evaluate_loss(dataset)
                rew = self.evaluate_reward(env)
                if self.logger:
                    self.logger.info(
                        f"Iteration: {step: d}, loss: {loss: .4f}, grad_norm :{grad_norm:.4f}, reward: {rew: .4f}."
                    )
                else:
                    print(
                        f"Iteration: {step: d}, loss: {loss: .4f}, grad_norm :{grad_norm:.4f}, reward: {rew: .4f}."
                    )

        rew = self.evaluate_reward(env)
        rew = float(rew)
        return rew

    def train_by_cvxpy(self, dataset: List[Transition], env: LinearBandit) -> float:
        pref_features, non_pref_features = [], []
        pref_ref_policy, non_pref_ref_policy = [], []
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

            act_prob = self.ref_policy(state)
            pref_ref_policy.append(act_prob[pref_act])
            non_pref_ref_policy.append(act_prob[non_pref_act])

        pref_features = np.stack(pref_features, axis=0)
        non_pref_features = np.stack(non_pref_features, axis=0)

        pref_ref_policy = np.stack(pref_ref_policy, axis=0)
        non_pref_ref_policy = np.stack(non_pref_ref_policy, axis=0)

        theta = cp.Variable(self.feature_dim)
        log_policy_diff = (non_pref_features - pref_features) @ theta
        log_ref_policy_diff = cp.log(non_pref_ref_policy) - cp.log(pref_ref_policy)

        tmp = self.reg_coef * (log_policy_diff - log_ref_policy_diff)

        loss = cp.sum(cp.logistic(tmp)) / len(dataset)
        problem = cp.Problem(cp.Minimize(loss))
        problem.solve(solver="ECOS", verbose=False)

        theta_arr = np.array(theta.value)

        self.param = theta_arr
        loss, reward = self.evaluate_loss(dataset), self.evaluate_reward(env)
        if self.logger:
            self.logger.info("Train by cvxopt.")
            self.logger.info(f"Loss calculated by cvxopt: {problem.value: .4f}.")
            self.logger.info(f"Loss: {loss: .4f}, reward: {reward: .4f}.")
        else:
            print("Train by cvxopt.")
            print(f"Loss calculated by cvxopt: {problem.value: .4f}.")
            print(f"Loss: {loss: .4f}, reward: {reward: .4f}.")

        return reward

    def evaluate_reward(self, env: LinearBandit) -> float:
        policy = self.ret_policy()
        rew = env.evaluate_reward(policy)

        return rew

    @property
    def get_param(self) -> np.ndarray:
        return self.param
