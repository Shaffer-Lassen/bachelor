#!/bin/bash

# Activate the virtual environment
source /home/mkv334/mde_env/bin/activate

# Go to the executing directory
cd /home/mkv334/csl/MDE02/pythonProject

# Run python training commands
echo "Predicting depth(s)"
python run.py \
	--image_path /home/mkv334/csl/MDE02/pythonProject/assets/example_images \
	--save_dir /home/mkv334/csl/MDE02/pythonProject/output \
	--checkpoint /home/mkv334/csl/MDE02/pythonProject/checkpoints/20250111_023801_scratch_epoch10.pth  \
	--side_by_side \
	--model scratch \
	--color_map spectral \
