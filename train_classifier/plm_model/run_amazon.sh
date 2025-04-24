task=amazon
gpu='0,1'
seed=0
train_seed=0
# model_type="bert-base-uncased"
model_type="bert-base-uncased"
prefix="attr"
seed=${seed}
train_seed=${train_seed}
max_seq_len=128
max_seq_len_test=128
eval_batch_size=256
steps=100
gen_model="gpt-3.5-turbo"
data_dir="../../dataset/${task}"
gen_model=${gen_model}_${prefix}
train_file="train/dado-gpt3-amazon.jsonl"
val_file="val/valid-amazon.jsonl"
test_file="test/test-amazon.jsonl"
cache_dir="../../cache"
lr=2e-5
batch_size=32
epochs=3
weight_decay=1e-3

output_dir="../../res_model/${task}"

train_cmd="python main.py --task=${task} \
	--train_file=${train_file} --val_file=${val_file} --test_file=${test_file} \
	--unlabel_file=unlabeled.json --tokenizer=${model_type} \
	--gen_model=${gen_model} --data_dir=${data_dir} --seed=${seed} --train_seed=${train_seed} \
	--cache_dir=${cache_dir} --output_dir=${output_dir}  \
	--gpu=${gpu} --num_train_epochs=${epochs} --weight_decay=${weight_decay} --learning_rate=${lr}  \
	--batch_size=${batch_size} --eval_batch_size=${eval_batch_size} \
	--max_seq_len=${max_seq_len} --max_seq_len_test=${max_seq_len_test} --auto_load=1 \
	--max_steps=${steps} --model_type=${model_type}"
echo $train_cmd
eval $train_cmd



