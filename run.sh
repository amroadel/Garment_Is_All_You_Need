#!/bin/bash

# Define the paths
MODEL_PATH="../results/body_images"
CLOTH_PATH="garment.jpg"
SCALE=2.0
SAMPLE=2
MODEL_TYPE="hd"
CATEGORY=0

declare -A garment_types
garment_types[0]="top"
garment_types[1]="bottom"
garment_types[2]="dress" 

GARMENT_TYPE="${garment_types[$CATEGORY]}"

python run.py --garment_path "$CLOTH_PATH" --garment_type "$GARMENT_TYPE"

cd OOTDiffusion/run
IMAGES_FOLDER="../../results/body_images"

# Loop through each image file in the folder
for IMAGE_PATH in "$IMAGES_FOLDER"/body*.png; do
    python run_ootd.py --model_path "$IMAGE_PATH" --cloth_path "../../$CLOTH_PATH" --scale "$SCALE" --sample "$SAMPLE"  --model_type "$MODEL_TYPE" --category $CATEGORY
done
