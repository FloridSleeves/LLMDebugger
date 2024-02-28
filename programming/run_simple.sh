dataset=$1
model=$2
output_dir=$3
strategy="simple"
python main.py \
  --run_name $output_dir \
  --root_dir ../output_data/$strategy/$dataset/$model/ \
  --dataset_path ../input_data/$dataset/dataset/probs.jsonl \
  --strategy $strategy \
  --model $model \
  --n_proc "1" \
  --testfile ../input_data/$dataset/test/tests.jsonl \
  --verbose \
  --port "8000"
