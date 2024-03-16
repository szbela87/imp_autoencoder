#!/bin/bash

num_sims=10
ff_opt=0
hidden_layer_num=5
start_layer=32

latent_num=8

input_num=8
neuron_num=120
shared_weight_num=0

family="v2"

for i in $(seq 1 $num_sims)
do
    echo "---------------"
    echo "Simulation: $i"
    echo "---------------"
    python simparam.py --id $i --family $family --hidden_layer_num $hidden_layer_num --start_layer $start_layer --latent_num $latent_num --input_num $input_num --neuron_num $neuron_num --shared_weight_num $shared_weight_num
    ./neural
done
