#!/bin/bash

set -e
set -o pipefail

cd /workspace/semantic-modeling

export PYTHONPATH=$(pwd)/pysm:$PYTHONPATH
export RUST_LOG="exec=debug,mira=debug,mira::assembling::models=debug,mira::assembling::models::mrr=trace,mira::assembling::searching::beam_search=trace,mira::assembling::auto_label=debug"
export RUST_BACKTRACE=1

rust_dir=/workspace/semantic-modeling/$1
shift 1

python -m pydebug.update_config \
    --config_file $rust_dir/rust-input.json \
    --config workdir:$rust_dir/rust

model_file=$(python -c "
from pathlib import Path
files = [file for file in Path('$rust_dir/rust/').iterdir() if file.name.startswith('model.') and file.name.endswith('.bin')]
assert len(files) <= 1
print(files[0].name if len(files) > 0 else '')")

cargo run --release -p exec -- -i $rust_dir/rust-input.json -c mira_config.yml exec_func $@ -m $rust_dir/rust/$model_file