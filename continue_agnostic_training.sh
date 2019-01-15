#!/bin/bash

#SBATCH -c 12
#SBATCH --mem=60G
#SBATCH --time=6-00:00:00
#SBATCH --gres=gpu:2

./train.py --original_lr_schedule --agnostic --resume --logdir train_log/maskrcnn_agnostic_from_scratch/ --load train_log/maskrcnn_agnostic_from_scratch/checkpoint
