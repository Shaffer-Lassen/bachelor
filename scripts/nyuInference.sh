# Activate the virtual enviroment

source /home/mkv334/mde_env/bin/activate

#Go to executing dir
cd /home/mkv334/csl/MDE02/pythonProject

python nyu_inference.py --checkpoint ./checkpoints/20250111_122852_scratch_transpose_epoch30.pth\
               --h5_file ./data/NYU.mat \
               --index 0-25 \
               --save_dir ./output \
               --model scratch_transpose
