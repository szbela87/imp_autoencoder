#!/bin/bash

num_sims=10
ff_opt=1

hidden_layer_num=7
start_layer=64
latent_num=16

input_num=117
neuron_num=474
shared_weight_num=0

family="v0"

load=1
directory=../results_nsl-kdd/$family/$family'_'$hidden_layer_num'_'$start_layer'_'$latent_num/

dataset="nsl-kdd"

for i in $(seq 1 $num_sims)
do
    echo "---------------"
    echo "Simulation: $i"
    echo "---------------"
    python simparam.py --id $i --family $family --hidden_layer_num $hidden_layer_num --start_layer $start_layer --latent_num $latent_num --input_num $input_num --neuron_num $neuron_num --shared_weight_num $shared_weight_num --dataset $dataset --load $load --directory $directory
    ./neural
done
