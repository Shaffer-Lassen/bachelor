

#This script trains the Resnet based model on three different encoder depth: Resnet50, ResNet101 and Resnet 152.
#Performance can be compared by running the eval.sh script on each of the generated .pth files.

cd ..

#Train models with default hyperparameters but different depths
python3 train.py --model res50 --epochs 10
python3 train.py --model res101 --epochs 10
python3 train.py --model res152 --epochs 10
