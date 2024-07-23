

# GPU_num=4
# for ((i=0;i<GPU_num;i++))
# CUDA_VISIBLE_DEVICES=${i} PORT=${i+11111}  bash ./tools/segmentation/dist_train.sh \
# configs/segmentation/tfa/nwpu/tfa_r101_fpn_nwpu-split1_10shot-fine-tuning.py  1 --seed ${i} 2>&1 

GPU_num=8
for ((i=0;i<GPU_num;i++))
do
    CUDA_VISIBLE_DEVICES=${i} PORT=$((11311+i)) bash ./tools/segmentation/dist_train.sh   configs/segmentation/tfa/nwpu/boss_r101_fpn_nwpu-split1_10shot-fine-tuning-resize-s4.py  1 --work-dir work_dirs/nwpu/finetune/boss_r101_fpn_nwpu-split1_5shot-bs8-fine-tuning-resize-s4-sephead-${i} --seed ${i} &
done

# GPU_num=4
# for ((i=0;i<GPU_num;i++)d)
# do
#     CUDA_VISIBLE_DEVICES=${i} PORT=$((11311+i)) bash ./tools/segmentation/dist_train.sh \
#     configs/segmentation/tfa/nwpu/tfa_r101_fpn_nwpu-split1_base-training_resize.py 1 --work-dir configs/segmentation/tfa/nwpu/tfa_r101_fpn_nwpu-split1_base-training_resize-bs8-s4-${i} --seed ${i} &
# done

# GPU_num=4
# for ((i=0;i<GPU_num;i++))
# do
#     CUDA_VISIBLE_DEVICES=$(($i+4)) PORT=$((11341+i)) bash ./tools/segmentation/dist_train.sh \
#     configs/segmentation/tfa/nwpu/tfa_r101_fpn_nwpu-split1_base-training_resize.py 1 --work-dir work_dirs/nwpu/base_training/tfa_r101_fpn_nwpu-split1_base-training_resize-bs2-s4-${i} --seed ${i} &
# done
