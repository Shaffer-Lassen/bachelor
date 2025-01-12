# Activate the virtual enviroment

source /home/mkv334/mde_env/bin/activate

#Go to executing dir
cd /home/mkv334/csl/MDE02/pythonProject

CUDA_DEVICE=0
LOSS=l1
EPOCHS=30
DATA=NYUv2
M1=
M2=swin
M3=scratch
M4=naive


#Run python training commands
#echo "Starting training Dino"
#python train.py --epochs $EPOCHS  --model $M1 --loss $LOSS --cuda $CUDA_DEVICE --dataset $DATA

#echo "Starting training ScratchModel"
#python train.py  --epochs $EPOCHS --model $M2 --loss $LOSS --cuda $CUDA_DEVICE  --dataset $DATA

#echo "Starting training DenseModel"
#python train.py --epochs $EPOCHS --model $M3 --loss $LOSS --cuda $CUDA_DEVICE --dataset $DATA --pretrained False

#echo "Starting training ResModel"
python train.py --epochs $EPOCHS --model $M4 --loss $LOSS --cuda $CUDA_DEVICE --dataset $DATA --pretrained False
