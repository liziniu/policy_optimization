#!/bin/bash

set -e
set -x

STATE_DIM=50
ACTION_NUM=10
PREF_DATA_NUM=50

for seed in 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030
do
    python -m experiments.run_neural_bandit \
    --state_dim ${STATE_DIM} \
    --action_num ${ACTION_NUM} \
    --pref_data_num ${PREF_DATA_NUM} \
    --rl_data_ratio 0.5 \
    --reg_coef 0.01 \
    --seed ${seed} \
    --logdir "log"
done
