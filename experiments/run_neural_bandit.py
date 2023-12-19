import argparse
import numpy as np
import torch.cuda
from torch.utils.tensorboard import SummaryWriter
import os

from copy import deepcopy

from algos.neural_bandit.mle import RewardModel, MaximumLikelihoodEstimator
from algos.neural_bandit.pg import (
    PolicyModel,
    PolicyGradientOptimizer,
    UniformPolicyModel,
)
from algos.neural_bandit.dpo import DirectPreferenceOptimizer
from envs.neural_bandit import NeuralBandit
from utils.io_utils import save_code, save_config, create_log_dir
from utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="neural_bandit")
    parser.add_argument("--state_dim", type=int, default=50)
    parser.add_argument("--action_num", type=int, default=10)

    parser.add_argument("--agent", type=str, default="pg")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--logdir", type=str, default="log")

    parser.add_argument("--pref_data_num", type=int, default=50)
    parser.add_argument("--mle_num_iters", type=int, default=20)

    parser.add_argument("--rl_data_ratio", type=float, default=4)
    parser.add_argument("--reg_coef", type=float, default=1.0)
    parser.add_argument("--pg_num_iters", type=int, default=50)

    return parser.parse_args()


def collect_preference_data(env, num_pref_data):
    datasets = []
    for i in range(num_pref_data):
        state = env.sample_states(n=1)
        action0, action1 = np.random.choice(env.action_num, size=2, replace=False)
        action0 = torch.tensor([int(action0)])
        action1 = torch.tensor([int(action1)])

        rew0 = env.get_state_action_reward(state, action0)[0]
        rew1 = env.get_state_action_reward(state, action1)[0]
        prob = torch.exp(rew0) / (torch.exp(rew0) + torch.exp(rew1))
        prob = prob.item()
        if np.random.random() < prob:
            datasets.append([state, action0, action1])
        else:
            datasets.append([state, action1, action0])
    return datasets


def main(args=parse_args()):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    log_dir = create_log_dir(args)
    save_code(log_dir)
    save_config(args.__dict__, log_dir)

    logger = Logger(log_dir)
    writer = SummaryWriter(log_dir)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    state_dim = args.state_dim
    action_num = args.action_num

    true_reward_model = RewardModel(
        state_dim,
        action_num,
        is_target=True,
        hidden_dim=64,
        num_layers=1,
        activation="tanh",
        device=device,
    )
    env = NeuralBandit(
        state_dim,
        action_num,
        true_reward_model,
        device=device,
        num_trials_for_eval=5000,
    )

    opt_policy = env.get_opt_policy()
    opt_rew = env.evaluate_policy(opt_policy)
    uniform_policy = UniformPolicyModel(action_num, device)
    uniform_policy_rew = env.evaluate_policy(uniform_policy)
    logger.info(
        f"Optimal reward: {opt_rew:.4f} uniform policy rew: {uniform_policy_rew:.4f}"
    )
    # Collect preference data
    pref_dataset = collect_preference_data(env, args.pref_data_num)

    learned_reward_model = RewardModel(
        state_dim,
        action_num,
        is_target=False,
        hidden_dim=64,
        num_layers=2,
        activation="relu",
        device=device,
    )

    mle_learner = MaximumLikelihoodEstimator(
        action_num,
        learned_reward_model,
        learning_rate=1e-3,
        batch_size=64,
        logger=logger,
    )
    states = torch.cat([x[0] for x in pref_dataset], dim=0)
    positive_actions = torch.cat([x[1] for x in pref_dataset], dim=0)
    negative_actions = torch.cat([x[2] for x in pref_dataset], dim=0)

    mle_learner.optimize(
        states, positive_actions, negative_actions, num_epochs=args.mle_num_iters
    )

    # Train policy on preference data
    logger.info("========Train on preference data [DPO]===========")
    policy = PolicyModel(
        state_dim,
        action_num,
        hidden_dim=64,
        num_layers=2,
        device=device,
    )
    policy2 = deepcopy(policy)
    policy2.load_state_dict(policy.state_dict())
    policy3 = deepcopy(policy)
    policy3.load_state_dict(policy.state_dict())

    learned_env = NeuralBandit(
        state_dim,
        action_num,
        learned_reward_model,
        num_trials_for_eval=5000,
        device=device,
    )

    dpo = DirectPreferenceOptimizer(
        policy,
        ref_policy=uniform_policy,
        env=env,
        learned_env=learned_env,
        learning_rate=1e-3,
        batch_size=64,
        beta=args.reg_coef,
        logger=logger,
    )
    states = torch.cat([x[0] for x in pref_dataset], dim=0)
    positive_actions = torch.cat([x[1] for x in pref_dataset], dim=0)
    negative_actions = torch.cat([x[2] for x in pref_dataset], dim=0)
    dpo.optimize(
        states,
        positive_actions,
        negative_actions,
        num_epochs=args.pg_num_iters,
        optimal_rew=opt_rew.item(),
    )
    dpo_reward = env.evaluate_policy(policy)
    logger.info("DPO reward error: {:.4f}".format(opt_rew - dpo_reward))

    # RMB-PO
    logger.info("========Train on preference data [RMB-PO]===========")
    pg = PolicyGradientOptimizer(
        policy2,
        ref_policy=uniform_policy,
        env=env,
        learned_env=learned_env,
        learning_rate=1e-3,
        batch_size=64,
        beta=args.reg_coef,
        logger=logger,
    )
    pg.optimize(
        states,
        positive_actions,
        negative_actions,
        num_epochs=args.pg_num_iters,
        optimal_rew=opt_rew.item(),
    )
    rmb_po_reward = env.evaluate_policy(policy2)
    logger.info("RMB PO reward error: {:.4f}".format(opt_rew - rmb_po_reward))

    # Train policy on additional RL data
    logger.info(
        f"========Train on {args.rl_data_ratio + 1}x RL data [RMB-PO+]==========="
    )
    pg2 = PolicyGradientOptimizer(
        policy3,
        ref_policy=uniform_policy,
        env=env,
        learned_env=learned_env,
        learning_rate=1e-3,
        batch_size=64,
        beta=args.reg_coef,
        logger=logger,
    )
    rl_states = env.sample_states(n=int(args.rl_data_ratio * args.pref_data_num))
    rl_states = torch.cat([rl_states, states], dim=0)

    pg2.optimize(rl_states, num_epochs=args.pg_num_iters, optimal_rew=opt_rew.item())

    rmb_po_plus_reward = env.evaluate_policy(policy3)
    logger.info("RMB PO+ reward error: {:.4f}".format(opt_rew - rmb_po_plus_reward))


if __name__ == "__main__":
    main()
