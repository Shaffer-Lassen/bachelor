source /home/mkv334/mde_env/bin/activate

#Go to executing dir
cd /home/mkv334/csl/MDE02/pythonProject

m1=naive
m2=scratch
m3=scratch
m4=scratch

m1_check=./checkpoints/20250111_171254_naive_epoch30.pth
m2_check=./checkpoints/20250110_094821_scratch_epoch30.pth
m3_check=./checkpoints/20250110_233200_scratch_epoch10.pth
m4_check=./checkpoints/20250110_225103_scratch_epoch10.pth

#echo "Evaluating scratch model"
python evaluate.py --checkpoint $m1_check --model $m1 
#echo "Evaluating Resnet model"
python evaluate.py --checkpoint $m2_check --model $m2 
#echo "Evaluating DenseNet model"
#python evaluate.py --checkpoint $m3_check --model $m3
#echo "Evaluating Dino model"
#python evaluate.py --checkpoint $m4_check --model $m4 
