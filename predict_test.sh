#!/usr/bin/env bash

IMDIR=sample_images # Dir containing your images
DEST=/tmp/proposals_test # Output files go here (jsons containing segm. masks)
mkdir -p $DEST

GPU="--gpu 0"
DEST="--forward $DEST"
IMG_DIR="--generic_images_folder $IMDIR"
IMG_PATTERN="--generic_images_pattern ./*.png"
DATASET_TYPE="--forward_dataset generic"
MODEL="--load train_log/maskrcnn_two_heads_from_scratch/model-854100.data-00000-of-00001" # Model trained on COCO
FLAGS="--second_head --agnostic" # Agnostic variant; Use second classification head

python train.py $GPU $DEST $IMG_DIR $IMG_PATTERN $DATASET_TYPE $MODEL $FLAGS

echo "Done."
