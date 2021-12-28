domain_list=('rest' 'laptop' 'service' 'device')
CUDA_VISIBLE_DEVICES='0'

# DS-BERT based on bert-e 

for domain in ${domain_list[@]};
do
	python ./ds-bert/run_mlm_modeling.py --model_name_or_path='./bert_lm_models/bert-e' \
		--train_data_file="./raw_data/${domain}_train.txt" \
		--eval_data_file="./raw_data/${domain}_train.txt" \
		--do_train \
		--do_eval \
		--output_dir="./bert_lm_models/ds-bert-e/${domain}"  
done
