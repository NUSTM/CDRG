#!/bin/bash
domains=('rest' 'service' 'laptop' 'device')
export CUDA_VISIBLE_DEVICES=0

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
	        python ./ds-bert/generate_pseudo.py --source_domain $src_domain \
	        	--target_domain $tar_domain \
	        	--model_name_or_path "./bert_lm_models/ds-bert-e/${tar_domain}" \
	        	--output_dir "./pseudo_output/ds-bert-e" \
                --batch_size 40
        fi
    done
done

