

#This script trains the Resnet based model on three different encoder depth: Resnet50, ResNet101 and Resnet 152.
#Performance can be compared by running the eval.sh script on each of the generated .pth files.

#Activate enviroment
source /home/mkv334/mde_env/bin/activate

#Go to executing dir
cd /home/mkv334/csl/MDE02/pythonProject

#Train models with default hyperparameters but different depths
python train.py --model res50 --epochs 10 --pretrained True --cuda 1
python train.py --model res101 --epochs 10 --pretrained True --cuda 1
python train.py --model res152 --epochs 10 --pretrained True --cuda 1
