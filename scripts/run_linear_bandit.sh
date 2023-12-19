#!/bin/bash

set -e
set -x

ACTION_NUM=4
PREF_DATA_NUM=20
PG_NUM_ITERS=1000
REG_COEF=0.01
STATE_DIM=1

for seed in 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030
do
    python -m experiments.run_linear_bandit \
    --mle_adaptive \
    --state_dim ${STATE_DIM} \
    --action_num ${ACTION_NUM} \
    --pref_data_num ${PREF_DATA_NUM} \
    --rl_data_ratio 0.5 \
    --pg_num_iters ${PG_NUM_ITERS} \
    --reg_coef ${REG_COEF} \
    --dpo_adaptive \
    --pg_adaptive \
    --seed ${seed} \
    --flip_feature \
    --logdir "log"
done