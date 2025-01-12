cd ..

echo "Predicting depth(s)"
python run.py \
	--image_path  \
	--save_dir \
	--checkpoint  \
	--side_by_side \
	--model \
	--color_map spectral \
