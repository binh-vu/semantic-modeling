#!/usr/bin/env bash

set -e
set -o pipefail

dataset="museum_edm"
declare -a kfolds=(
"{\"train_sm_ids\": [\"s01-s14\"], \"test_sm_ids\": [\"s15-s28\"]}"
"{\"train_sm_ids\": [\"s15-s28\"], \"test_sm_ids\": [\"s01-s14\"]}"
"{\"train_sm_ids\": [\"s08-s21\"], \"test_sm_ids\": [\"s01-s07\", \"s22-s28\"]}"
)
declare stypes=(
"SereneSemanticType"
"MohsenJWS"
"ReImplMinhISWC"
)

for stype in "${stypes[@]}"; do

    exp_name=$(python -c "from datetime import datetime; print(datetime.now().strftime('%B_%d__%H:%M:%S').lower());")
    exp_dir="$(pwd)/debug/experiments/${dataset}_${exp_name}"
    commit_id=$(git rev-parse HEAD)

    if [ -f $exp_dir ]; then
        echo "ExperimentDirectory should not exists!"
        exit -1
    fi

    mkdir -p $exp_dir
    echo $commit_id > "$exp_dir/commit_id.txt"
    echo "Execution dir: $exp_dir"
    echo "" > "$exp_dir/execution.log"
    echo "" > $(pwd)/debug/execution.log

    for kfold in "${kfolds[@]}"; do
        PYTHONPATH=$(pwd)/pysm python -m experiments.semantic_labeling.evaluation \
            --dataset=$dataset \
            --kfold="$kfold" \
            --semantic_typer=$stype \
            --exp_dir="$exp_dir"
    done
done