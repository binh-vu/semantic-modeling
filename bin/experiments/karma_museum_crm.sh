declare -a kfolds=(
"{\"train_sm_ids\": [\"s01-s14\"], \"test_sm_ids\": [\"s15-s28\"]}"
"{\"train_sm_ids\": [\"s15-s28\"], \"test_sm_ids\": [\"s01-s14\"]}"
"{\"train_sm_ids\": [\"s08-s21\"], \"test_sm_ids\": [\"s01-s07\", \"s22-s28\"]}"
)
dataset="museum_crm"

declare -a conf_sm_typers=(
"MohsenJWS"
)
declare -a conf_n_stypes=(1)
n_conf=${#conf_n_stypes[@]}

. ./bin/experiments/func/karma_func.sh

for (( i=0; i<${n_conf}; i++ )); do
    n_stypes=${conf_n_stypes[$i]}
    styper=${conf_sm_typers[$i]}

    exec_karma $dataset $styper $n_stypes "sm_karma_$dataset" "${kfolds[@]}"
#    break
done