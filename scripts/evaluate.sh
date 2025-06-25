# set -x
model=${1}
step=${2}

output_dir=outputs/$model
mkdir -p $output_dir/wavs

if [ $step -le 1 ]; then
    python main.py --model $model --result_dir $output_dir
fi

if [ $step -le 2 ]; then
    python3 -m tools.api_judge --src_file $output_dir/${model}_results.jsonl --tgt_file $output_dir/${model}_results_judge.jsonl 
fi

if [ $step -le 3 ]; then
    python3 -m tools.retrieve_results --src_file $output_dir/${model}_results_judge.jsonl
fi