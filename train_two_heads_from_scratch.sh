#!/bin/bash

#SBATCH -c 12
#SBATCH --mem=60G
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:2

./train.py --load train_log/fastrcnn/COCO-ResNet101-MaskRCNN.npz --original_lr_schedule --agnostic --second_head --logdir train_log/maskrcnn_two_heads_from_scratch/