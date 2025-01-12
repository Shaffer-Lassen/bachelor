cd ..

model=naive
checkpoint=./checkpoints/{checkpoint}


echo "Evaluating model"
python evaluate.py --model $model --checkpoint $checkpoint  

