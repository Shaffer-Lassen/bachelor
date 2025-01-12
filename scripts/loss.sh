# Activate the virtual enviroment

source /home/mkv334/mde_env/bin/activate

#Go to executing dir
cd /home/mkv334/csl/MDE02/pythonProject

CUDA_DEVICE=1
EPOCHS=10
DATA=NYUv2
M1=scratch

python train.py --epochs $EPOCHS  --model $M1 --loss l1 --cuda $CUDA_DEVICE --dataset $DATA --pretrained True
python train.py  --epochs $EPOCHS --model $M1 --loss l1smooth --cuda $CUDA_DEVICE  --dataset $DATA --pretrained True
python train.py --epochs $EPOCHS --model $M1 --loss l2 --cuda $CUDA_DEVICE --dataset $DATA --pretrained True
python train.py --epochs $EPOCHS --model $M1 --loss huber --cuda $CUDA_DEVICE --dataset $DATA --pretrained True
