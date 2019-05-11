#!/bin/bash
set -e
set -o pipefail

exec_karma () {

    dataset=$1
    semantic_labeling=$2
    top_n_stypes=$3
    pyexp_name=$4
    shift 4
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

    python -m experiments.clear_cache --dataset=$dataset

    for kfold in "${kfolds[@]}"; do
        echo ">>>>>> semantic-labeling-method=$semantic_labeling, #TYPES=$top_n_stypes, KFOLD = $kfold"
        echo "Execute command:
            python -m experiments.semantic_modeling.kfold_karma \\
                --dataset=$dataset \\
                --semantic_typer=$semantic_labeling \\
                --semantic_labeling_top_n_stypes=$top_n_stypes \\
                --kfold=\"$kfold\" \\
                --exp_dir=$exp_dir 2>&1 | tee -a \"$exp_dir/execution.log\" $workdir/debug/execution.log"
                
        python -m experiments.semantic_modeling.kfold_karma \
            --dataset=$dataset \
            --semantic_typer=$semantic_labeling \
            --semantic_labeling_top_n_stypes=$top_n_stypes \
            --kfold="$kfold" \
            --exp_dir=$exp_dir 2>&1 | tee -a "$exp_dir/execution.log" $workdir/debug/execution.log
    done

#    echo "Executing command:
#        python -m experiments.semantic_modeling.kfold_record \\
#        --dataset $dataset \\
#        --run_name $semantic_labeling#$top_n_stypes \\
#        --exp_name $pyexp_name \\
#        --exp_dir $exp_dir" | tee -a "$exp_dir/execution.log" $workdir/debug/execution.log
#
#    python -m experiments.semantic_modeling.kfold_record \
#        --dataset=$dataset \
#        --run_name="$semantic_labeling#$top_n_stypes" \
#        --exp_name="$pyexp_name" \
#        --exp_dir=$exp_dir 2>&1 | tee -a "$exp_dir/execution.log" $workdir/debug/execution.log
}