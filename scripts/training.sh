cd ..
CUDA_DEVICE=0
LOSS=l1
EPOCHS=30
DATA=NYUv2
MODEL=scratch

python3 train.py --epochs $EPOCHS --model $MODEL --loss $LOSS --cuda $CUDA_DEVICE --dataset $DATA --pretrained False
