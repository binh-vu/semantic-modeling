#!/bin/bash

export RUST_LOG="mira=debug,mira::assembling::models=debug,mira::assembling::searching::search_discovery:debug"

declare -a kfolds=(
"{\"train_sm_ids\": [\"s01-s14\"], \"test_sm_ids\": [\"s15-s28\"]}"
"{\"train_sm_ids\": [\"s15-s28\"], \"test_sm_ids\": [\"s01-s14\"]}"
"{\"train_sm_ids\": [\"s08-s21\"], \"test_sm_ids\": [\"s01-s07\", \"s22-s28\"]}"
)
dataset="museum_edm"
simulate_testing="false"
is_clear_cache="false"
workdir=$(pwd)

stype=$1
if [ "$stype" == "" ]; then
    stype="ReImplMinhISWC"
fi

. ./bin/experiments/func/mira_func.sh
exec_mira $dataset $stype 4 $simulate_testing $is_clear_cache "sm_mira_$dataset" "${kfolds[@]}"