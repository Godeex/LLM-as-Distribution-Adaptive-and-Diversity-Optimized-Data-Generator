# python ./gen_train_data/dk_gen_attrprompt.py --dataset 'amazon' --n_sample 50 --batch_size 05 --model_name 'deepseek-v3' --output_dir ./res
# python ./gen_train_data/dk_gen_attrprompt.py --dataset 'amazon' --n_sample 50 --batch_size 50 --model_name 'gpt-3.5-turbo' --output_dir ./res
# python ./gen_train_data/gen_attrprompt_v_1.py --dataset 'amazon' --n_sample 20 --batch_size 20 --model_name 'gpt-3.5-turbo' --output_dir ./res
# python ./gen_train_data/gen_attrprompt_v_1.py --dataset 'amazon' --n_sample 50 --batch_size 50 --model_name 'deepseek-v3' --output_dir ./res
# python ./gen_train_data/gen_attrprompt_v_2.py --dataset 'amazon' --n_sample 30 --batch_size 30 --model_name 'gpt-3.5-turbo' --output_dir ./res
# python ./gen_train_data/gen_attrprompt_v_2.py --dataset 'sst-2' --n_sample 10 --batch_size 10 --model_name 'deepseek-v3' --output_dir ./res --sample_dir ./dataset/sst-2/train/train-sst2.jsonl
# python ./gen_train_data/gen_attrprompt_v_3.py --dataset 'reddit' --n_sample 30 --batch_size 10 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/reddit/train/train-reddit-sample.jsonl
# python ./gen_train_data/gen_attrprompt_v_4.py --dataset 'reddit' --n_sample 30 --batch_size 10 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/reddit/train/train-reddit-sample.jsonl
# python ./gen_train_data/gen_attrprompt_v_4.py --dataset 'sst-2' --n_sample 30 --batch_size 30 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/sst-2/train/train-sst2.jsonl
# python ./gen_train_data/gen_attrprompt_v_4.py --dataset 'amazon' --n_sample 40 --batch_size 20 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/amazon/train/train-amazon-choice.jsonl
# python ./gen_train_data/gen_attrprompt_v_5.py --dataset 'reddit' --n_sample 40 --batch_size 20 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/reddit/train/train-reddit-choice.jsonl
# python ./gen_train_data/gen_attrprompt_v_5.py --dataset 'reddit' --n_sample 40 --batch_size 20 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/reddit/train/train-reddit.jsonl
# python ./gen_train_data/gen_attrprompt_v_5.py --dataset 'sst-2' --n_sample 40 --batch_size 20 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/sst-2/train/train-sst2-choice.jsonl
# python ./gen_train_data/gen_attrprompt_v_5.py --dataset 'sst-2' --n_sample 40 --batch_size 20 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/sst-2/train/train-sst2-sort.jsonl



# 40
# python ./gen_train_data/gen_attrprompt_v_6.py --dataset 'amazon' --n_sample 40 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/amazon/train/train-amazon-choice.jsonl
# python ./gen_train_data/gen_simprompt_v_6.py --dataset 'amazon' --n_sample 40 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/amazon/train/train-amazon-choice.jsonl
# python ./gen_train_data/gen_attrprompt_v_6.py --dataset 'amazon' --n_sample 40 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res_exp --sample_dir ./dataset/amazon/train/train-amazon-choice.jsonl
# python ./gen_train_data/gen_attrprompt_v_6.py --dataset 'amazon' --n_sample 40 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res_exp --sample_dir ./dataset/amazon/train/train-amazon-choice.jsonl


# python ./gen_train_data/gen_attrprompt_v_6.py --dataset 'reddit' --n_sample 40 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/reddit/train/train-reddit-choice.jsonl
# python ./gen_train_data/gen_simprompt_v_6.py --dataset 'reddit' --n_sample 40 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/reddit/train/train-reddit-choice.jsonl
# python ./gen_train_data/gen_attrprompt_v_6.py --dataset 'reddit' --n_sample 40 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res_exp --sample_dir ./dataset/reddit/train/train-reddit-choice.jsonl

# 100
# python ./gen_train_data/gen_attrprompt_v_6.py --dataset 'amazon' --n_sample 100 --batch_size 25 --model_name 'gpt-3.5-turbo' --output_dir ./res_exp_100 --sample_dir ./dataset/amazon/train/train-amazon-choice.jsonl
# python ./gen_train_data/gen_attrprompt_v_6.py --dataset 'reddit' --n_sample 100 --batch_size 25 --model_name 'gpt-3.5-turbo' --output_dir ./res_exp_100 --sample_dir ./dataset/reddit/train/train-reddit-choice.jsonl

# 400
# python ./gen_train_data/gen_attrprompt_v_6.py --dataset 'sst-2' --n_sample 400 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res_exp_100 --sample_dir ./dataset/sst-2/train/train-sst2-choice.jsonl



# generate preset attribute value
# python ./gen_train_data/gpt_gen_attr_value.py --dataset 'reddit' --attribute 'experience' --model_name 'gpt-3.5-turbo' --n_sample 10 --output_dir ./res
# generate random attribute name
# python ./gen_train_data/gpt_gen_attr_name_v1.py --dataset 'reddit' --model_name 'gpt-3.5-turbo' --n_sample 10 --output_dir ./res --sample_dir ./dataset/reddit/train/train-reddit-cluster.jsonl
# filter attribute name
# python ./gen_train_data/gpt_gen_attr_filter.py --dataset 'reddit' --model_name 'deepseek-v3' --n_sample 10 --output_dir ./res


# python ./gen_train_data/gpt_gen_attrprompt_multi_label_new.py --dataset 'arxiv' --n_sample 3 --output_dir ./res
# python ./gen_train_data/gpt_gen_simprompt_new.py --dataset 'amazon' --n_sample 5 --output_dir ./res
# python ./gen_train_data/filter_new.py --dataset 'amazon' --attribute 'product_name' --output_dir ./dataset_filter

# python ./gen_train_data/gen_attr_w_v2.py --dataset 'reddit' --model_name 'gpt-3.5-turbo' --n_sample 5 --output_dir ./res --sample_dir ./dataset/reddit/train/train-reddit-sample.jsonl
# python ./gen_train_data/gen_attr_w_v3.py --dataset 'sst-2' --model_name 'gpt-3.5-turbo' --n_sample 100 --output_dir ./res --sample_dir ./dataset/sst-2/train/train-sst2-cluster.jsonl
# python ./gen_train_data/gen_attr_w_v3.py --dataset 'amazon' --model_name 'gpt-3.5-turbo' --n_sample 100 --output_dir ./res --sample_dir ./dataset/amazon/train/train-amazon-cluster.jsonl
# python ./gen_train_data/gen_attr_w_v3.py --dataset 'reddit' --model_name 'gpt-3.5-turbo' --n_sample 100 --output_dir ./res --sample_dir ./dataset/reddit/train/train-reddit-cluster.jsonl

# Exp data
# python ./gen_train_data/gen_attrprompt_v_5.py --dataset 'amazon' --n_sample 100 --batch_size 25 --model_name 'gpt-3.5-turbo' --output_dir ./res_exp --sample_dir ./dataset/amazon/train/train-amazon-choice.jsonl
# python ./gen_train_data/gen_attrprompt_v_5.py --dataset 'reddit' --n_sample 100 --batch_size 25 --model_name 'gpt-3.5-turbo' --output_dir ./res_exp --sample_dir ./dataset/reddit/train/train-reddit-choice.jsonl
# python ./gen_train_data/gen_attrprompt_v_5.py --dataset 'sst-2' --n_sample 300 --batch_size 30 --model_name 'gpt-3.5-turbo' --output_dir ./res_exp --sample_dir ./dataset/sst-2/train/train-sst2-choice.jsonl

# python ./gen_train_data/gen_attr_w_v3.py --dataset 'reddit' --model_name 'gpt-3.5-turbo' --n_sample 40 --output_dir ./res_w/gold --sample_dir ./dataset/reddit/train/train-reddit-class0-sample.jsonl
# python ./gen_train_data/gen_attr_w_v3.py --dataset 'reddit' --model_name 'gpt-3.5-turbo' --n_sample 40 --output_dir ./res_w/v6 --sample_dir ./dataset/reddit/train/models/v6-gpt3-reddit-class0-sample-choice-100.jsonl
# python ./gen_train_data/gen_attr_w_v3.py --dataset 'reddit' --model_name 'gpt-3.5-turbo' --n_sample 40 --output_dir ./res_w/sim --sample_dir ./dataset/reddit/train/simprompt-reddit-class0-100.jsonl


# python ./gen_train_data/gen_attr_w_v3.py --dataset 'sst-2' --model_name 'gpt-3.5-turbo' --n_sample 40 --output_dir ./res_w/gold --sample_dir ./dataset/sst-2/train/train-sst2-class0-100.jsonl
# python ./gen_train_data/gen_attr_w_v3.py --dataset 'sst-2' --model_name 'gpt-3.5-turbo' --n_sample 40 --output_dir ./res_w/v6 --sample_dir ./res_exp_100/train/sst-2/sst2-gpt3-100-v6-class0.jsonl
# python ./gen_train_data/gen_attr_w_v3.py --dataset 'sst-2' --model_name 'gpt-3.5-turbo' --n_sample 40 --output_dir ./res_w/sim --sample_dir ./dataset/sst-2/train/simprompt-sst2-100-class0.jsonl


# python ./gen_train_data/gen_attrprompt_v_6a.py --dataset 'amazon' --n_sample 40 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/amazon/train/train-amazon-choice.jsonl
# python ./gen_train_data/gen_attrprompt_v_6a.py --dataset 'reddit' --n_sample 40 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/reddit/train/train-reddit-choice.jsonl
# python ./gen_train_data/gen_attrprompt_v_6a.py --dataset 'sst-2' --n_sample 40 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/sst-2/train/train-sst2-choice.jsonl


# python ./gen_train_data/gen_attrprompt_v_6a.py --dataset 'amazon' --n_sample 40 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/amazon/train/train-amazon.jsonl
# python ./gen_train_data/gen_attrprompt_v_6a.py --dataset 'reddit' --n_sample 40 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/reddit/train/train-reddit.jsonl
# python ./gen_train_data/gen_attrprompt_v_6a.py --dataset 'sst-2' --n_sample 40 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/sst-2/train/train-sst2.jsonl


# python ./gen_train_data/gen_attrprompt_v_6b.py --dataset 'amazon' --n_sample 40 --batch_size 40 --model_name 'gpt-4' --output_dir ./res --sample_dir ./dataset/amazon/train/train-amazon-choice.jsonl
# python ./gen_train_data/gen_attrprompt_v_6b.py --dataset 'amazon' --n_sample 40 --batch_size 40 --model_name 'deepseek-v3' --output_dir ./res --sample_dir ./dataset/amazon/train/train-amazon-choice.jsonl
# python ./gen_train_data/gen_attrprompt_v_6b.py --dataset 'reddit' --n_sample 40 --batch_size 40 --model_name 'deepseek-v3' --output_dir ./res --sample_dir ./dataset/reddit/train/train-reddit-choice.jsonl
python ./gen_train_data/gen_attrprompt_v_6.py --dataset 'sst-2' --n_sample 40 --batch_size 40 --model_name 'gpt-3.5-turbo' --output_dir ./res --sample_dir ./dataset/sst-2/train/train-sst2-choice.jsonl
