#!/bin/sh
PARTITION=Segmentation

  
GPU_ID=0
dataset=iSAID # iSAID LoveDA
arch=MVLPNet   

variable1= None
variable2= None

for net in resnet50  # vgg resnet50
do
        for shot in 1 # 1 5 
        do
                for split in  1   # 0 1 2
                do
                        exp_dir=exp/${arch}/${dataset}/${net}/split${split} # 
                        snapshot_dir=${exp_dir}/${shot}shot
                        result_dir=${exp_dir}/result
                        mkdir -p ${snapshot_dir} ${result_dir}
                        now=$(date +"%Y%m%d_%H%M%S")

                        echo ${arch}_${dataset}
                        echo ${net}_split${split}_${shot}shot

                        CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u train.py \
                                                --arch=${arch} \
                                                --shot=${shot} \
                                                --split=${split} \
                                                --backbone=${net} \
                                                --dataset=${dataset} \
                                                --variable1=${variable1} \
                                                --variable2=${variable2} \
                                                2>&1 | tee ${result_dir}/train-$now.log        
                done
        done
done
                        
                        
                        
                        
                        
                        # CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch --nproc_per_node=4 train.py \