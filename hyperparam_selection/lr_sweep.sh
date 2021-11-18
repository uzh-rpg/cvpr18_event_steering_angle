#!/bin/bash

TRAIN_DIR=/media/HHD/ana/dvs_steering_learning/DAVIS_small_new/training/
VAL_DIR=/media/HHD/ana/dvs_steering_learning/DAVIS_small_new/validation/
TEST_DIR=/media/HHD/ana/dvs_steering_learning/DAVIS_small_new/testing/

N_exp=5
min_lr=0.00005
max_lr=0.0005

step=$(bc -l <<< "($max_lr - $min_lr) /($N_exp - 1)")


for lr in `seq $min_lr $step $max_lr`
do
	exp_rootdir=/media/HHD/ana/dvs_steering_learning/models/hyper_test/expr_lr_$lr

	# Train
	python3.4 ../cnn.py --experiment_rootdir=$exp_rootdir \
 	--train_dir=$TRAIN_DIR --val_dir=$VAL_DIR --frame_mode=dvs \
	--initial_lr=$lr --epochs=30 --norandom_seed

	# Test
	python3.4 ../evaluation.py --experiment_rootdir=$exp_rootdir --test_dir=$TEST_DIR \
		--frame_mode=dvs --weights_fname=model_weights_29.h5

done
