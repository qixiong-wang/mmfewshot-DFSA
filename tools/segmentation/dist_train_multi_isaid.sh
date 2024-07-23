

# GPU_num=4
# for ((i=0;i<GPU_num;i++))
# CUDA_VISIBLE_DEVICES=${i} PORT=${i+11111}  bash ./tools/segmentation/dist_train.sh \
# configs/segmentation/tfa/nwpu/tfa_r101_fpn_nwpu-split1_10shot-fine-tuning.py  1 --seed ${i} 2>&1 

# GPU_num=8
# for ((i=0;i<GPU_num;i++))
# do
#     CUDA_VISIBLE_DEVICES=${i} PORT=$((11311+i)) bash ./tools/segmentation/dist_train.sh \
#     configs/segmentation/tfa/nwpu/tfa_r101_fpn_nwpu-split1_10shot-fine-tuning.py  1 --work-dir work_dirs/tfa_r101_fpn_nwpu-split1_10shot-fine-tuning-lr2-${i} --seed ${i} &
# done

GPU_num=4
for ((i=0;i<GPU_num;i+=2))
do
    CUDA_VISIBLE_DEVICES=$(($i +4)),$(($i + 5)) PORT=$((11331+i)) bash ./tools/segmentation/dist_train.sh  configs/isaid/r101_fpn_FSD_isaid-split1_base-training.py 2 --work-dir work_dirs/isaid/base_training/split1/r101_fpn_fsd_isaid-split1_base-training-${i} --seed ${i} &
done

# CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=12555 bash ./tools/segmentation/dist_train.sh configs/segmentation/tfa/isaid/tfa_r101_fpn_isaid-split3_base-training_resize.py 4 --work-dir work_dirs/tfa_r101_fpn_isaid-split3_base_training-resize-lr0.005-0 --seed 0 

# GPU_num=4
# for ((i=0;i<GPU_num;i++))
# do
#     CUDA_VISIBLE_DEVICES=$(($i )) PORT=$((11351+i)) bash ./tools/segmentation/dist_train.sh \
#     configs/segmentation/tfa/isaid/tfa_r101_fpn_isaid-split3_base-training_resize_16w_bs8.py 1 --work-dir work_dirs/tfa_r101_fpn_isaid-split3_base_training-resize-1gpu-16w-bs8-lr0.005-${i} --seed ${i} &
# done



# GPU_num=4
# for ((i=0;i<GPU_num;i++))
# do
#     CUDA_VISIBLE_DEVICES=$(($i+4)) PORT=$((11331+i)) bash ./tools/segmentation/dist_train.sh \
#     configs/segmentation/tfa/isaid/tfa_r101_fpn_isaid-split1_base-training_resize_16w_bs8.py  1 --work-dir work_dirs/tfa_r101_fpn_isaid-split1_base_training-resize-1gpu-16w-bs8-lr0.005-${i} --seed ${i} &
# done

# GPU_num=4
# for ((i=0;i<GPU_num;i++))
# do
#     CUDA_VISIBLE_DEVICES=${i} PORT=$((11327+i)) bash ./tools/segmentation/dist_train.sh configs/segmentation/tfa/isaid/boss_r101_fpn_isaid-split2_1shot-fine-tuning_resize.py 1  --work-dir work_dirs/isaid/finetune/boss_r101_fpn_isaid-split2_5shot-fine-tuning_resize-${i} --seed ${i} &
# done


