dataset=$1
model=$2
seedfile=$3
output_dir=$4
strategy="ldb"
python main.py \
  --run_name $output_dir \
  --root_dir ../output_data/$strategy/$dataset/$model/ \
  --dataset_path ../input_data/$dataset/dataset/probs.jsonl \
  --strategy $strategy \
  --model $model \
  --seedfile $seedfile \
  --pass_at_k "1" \
  --max_iters "10" \
  --n_proc "1" \
  --port "8000" \
  --testfile ../input_data/$dataset/test/tests.jsonl \
  --verbose
