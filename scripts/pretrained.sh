source /home/mkv334/mde_env/bin/activate

#Go to executing dir
cd /home/mkv334/csl/MDE02/pythonProject

CUDA_DEVICE=0
LOSS=l1
EPOCHS=30
DATA=NYUv2


M1=dino
M2=res50
M3=dense
M4=swin




python train.py --epochs $EPOCHS  --model $M1 --loss $LOSS --cuda $CUDA_DEVICE --dataset $DATA --pretrained True

python train.py  --epochs $EPOCHS --model $M2 --loss $LOSS --cuda $CUDA_DEVICE  --dataset $DATA --pretrained True

python train.py --epochs $EPOCHS --model $M3 --loss $LOSS --cuda $CUDA_DEVICE --dataset $DATA --pretrained True

python train.py --epochs $EPOCHS --model $M4 --loss $LOSS --cuda $CUDA_DEVICE --dataset $DATA --pretrained True
