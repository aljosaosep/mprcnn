#!/bin/bash

#SBATCH -c 12
#SBATCH --mem=60G
#SBATCH --time=6-00:00:00
#SBATCH --gres=gpu:2

./train.py --load /fastwork/${USER}/mywork/data/pretrained_models/COCO-ResNet101-MaskRCNN.npz --original_lr_schedule --agnostic --logdir train_log/maskrcnn_agnostic_from_scratch/
