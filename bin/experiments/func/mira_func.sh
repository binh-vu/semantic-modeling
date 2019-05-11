#!/bin/bash
set -e
set -o pipefail

exec_mira () {
    dataset=$1
    semantic_labeling=$2
    top_n_stypes=$3
    simulate_testing=$4
    is_clear_cache=$5
    pyexp_name=$6
    shift 6
    kfolds=("$@")

    workdir="$(pwd)"
    export PYTHONPATH="$PYTHONPATH:$workdir/pysm"
    
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

    if [ "$is_clear_cache" = "true" ]; then
        echo "Clear cache..."
        python -m experiments.clear_cache --dataset=$dataset
    fi

    for kfold in "${kfolds[@]}"; do
        echo ">>>>>> semantic-labeling-method=$semantic_labeling, #TYPES=$top_n_stypes, KFOLD = $kfold"
        echo "Execute: python -m experiments.semantic_modeling.kfold_mira \
            --dataset=$dataset \
            --func="gen_input" \
            --styper=$semantic_labeling \
            --styper_top_n_stypes=$top_n_stypes \
            --styper_simulate_testing=$simulate_testing \
            --kfold="$kfold" \
            --exp_dir=$exp_dir"
        python -m experiments.semantic_modeling.kfold_mira \
            --dataset=$dataset \
            --func="gen_input" \
            --styper=$semantic_labeling \
            --styper_top_n_stypes=$top_n_stypes \
            --styper_simulate_testing=$simulate_testing \
            --kfold="$kfold" \
            --exp_dir=$exp_dir 2>&1 | tee -a "$exp_dir/execution.log" $workdir/debug/execution.log

        rust_dir="kfold-$(python -m experiments.arg_helper --dataset $dataset --kfold "$kfold" --func get_short_train_name)"
        cargo run --release \
            -p exec -- \
            -i $exp_dir/$rust_dir/rust-input.json \
            -c $workdir/mira_config.yml \
            exp -g "false" 2>&1 | tee -a "$exp_dir/execution.log" $workdir/debug/execution.log

        echo "Executing command:
        python -m experiments.semantic_modeling.kfold_mira
            --dataset=$dataset
            --func=handle_output
            --config_file=$workdir/mira_config.yml
            --kfold=$kfold
            --exp_dir=$exp_dir" | tee -a "$exp_dir/execution.log" $workdir/debug/execution.log

        python -m experiments.semantic_modeling.kfold_mira \
            --dataset=$dataset \
            --func="handle_output" \
            --config_file="$workdir/mira_config.yml" \
            --kfold="$kfold" \
            --exp_dir=$exp_dir 2>&1 | tee -a "$exp_dir/execution.log" $workdir/debug/execution.log
    done

#    echo "Executing command:
#        python -m experiments.semantic_modeling.kfold_record
#        --dataset $dataset
#        --run_name $semantic_labeling#$top_n_stypes
#        --exp_name sm_mira_$dataset
#        --exp_dir $exp_dir" | tee -a "$exp_dir/execution.log" $workdir/debug/execution.log
#
#    python -m experiments.semantic_modeling.kfold_record \
#        --dataset $dataset \
#        --exp_name $pyexp_name \
#        --exp_dir $exp_dir 2>&1 | tee -a "$exp_dir/execution.log" $workdir/debug/execution.log
}
