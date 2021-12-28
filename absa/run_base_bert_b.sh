#!/bin/bash
domains=('rest' 'service' 'laptop' 'device')
export CUDA_VISIBLE_DEVICES=0
output='./run_out/ds-bert-b/'

for tar_domain in ${domains[@]};
do
    for src_domain in  ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then
            if [ $src_domain == 'laptop' -a  $tar_domain == 'device' ];
            then
                continue
            fi
            if [ $src_domain == 'device' -a  $tar_domain == 'laptop' ];
            then
                continue
            fi
            echo "${src_domain}-${tar_domain}"
	        python ./absa/run_bert_absa.py --task_type 'absa' \
                --data_dir "./pseudo_output/ds-bert-b/#${src_domain}-${tar_domain}#merge"  \
                --output_dir "${output}${src_domain}-${tar_domain}"  \
                --train_batch_size 16 \
                --bert_model 'bert_base' \
                --seed 43 \
                --do_train \
                --do_eval 
        fi
    done
done



