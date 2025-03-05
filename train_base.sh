#!/bin/sh
PARTITION=Segmentation

    
GPU_ID=3
GPU_num=1
dataset=iSAID # iSAID iSAID_1

arch=PSPNet #PSPNet_me
net=resnet50 # vgg resnet50 resnet101
variable1=None
variable2=None

for split in  2   # 0 1 2
do
        exp_dir=initmodel/${arch}/${dataset}/split${split}/${net} # 
        snapshot_dir=${exp_dir}
        result_dir=${exp_dir}/result
        mkdir -p ${snapshot_dir} ${result_dir}
        now=$(date +"%Y%m%d_%H%M%S")
        echo ${arch}_${dataset}
        echo ${net}_split${split}

        CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u train_base.py \
        --arch=${arch} \
        --split=${split} \
        --backbone=${net} \
        --dataset=${dataset} \
        --variable1=${variable1} \
        --variable2=${variable2} \
        2>&1 | tee ${result_dir}/train_base-$now.log        

done