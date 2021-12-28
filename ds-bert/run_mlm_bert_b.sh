domain_list=('rest' 'laptop' 'service' 'device')
CUDA_VISIBLE_DEVICES='1'

# DS-BERT based on bert-b 

for domain in ${domain_list[@]};
do
	python ./ds-bert/run_mlm_modeling.py --model_name_or_path='bert-base-uncased' \
		--train_data_file="./raw_data/${domain}_train.txt" \
		--eval_data_file="./raw_data/${domain}_train.txt" \
		--do_train \
		--do_eval \
		--output_dir="./bert_lm_models/ds-bert-b/${domain}"  
done
