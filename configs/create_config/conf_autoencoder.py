"""
Implicit configuration generator with undirected latent layer.
Parameters:
- hidden_layer_num: number of the hidden layers
- start_layer: the size of the first layer in the encoder
- latent_num: the size of the latent layer
- input_num: the size of the input
- act_type: activation function type 3 = Leaky Relu, 9 = arctan.
- family: model family: v0, v1, v2

Examples:
    Conf      Call
   5;32;8    python conf_autoencoder.py --hidden_layer_num 5 --start_layer 32 --latent_num 8 --input_num 8 --act_type 9 --family v0
"""

import argparse
from utils import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_layer_num',type=int,default=5)
parser.add_argument('--start_layer',type=int,default=64)
parser.add_argument('--latent_num',type=int,default=8)
parser.add_argument('--input_num',type=int,default=8)
parser.add_argument('--act_type',type=int,default=9) 
parser.add_argument('--family',type=str,default="v0") 

args = parser.parse_args()

hidden_layer_num = args.hidden_layer_num # Hidden layers
start_layer = args.start_layer # Size of the first encoder layer
latent_num = args.latent_num # Latent layer size
input_num = args.input_num # Input size - also the output size
act_type = args.act_type # Activation type
family = args.family # model family

output_num = input_num

encoder = [input_num]
for i in range(hidden_layer_num//2):
    encoder.append(start_layer//2**i)
    
decoder = encoder[::-1] # reversed encoder
    
layers = encoder + [latent_num] + decoder # the configuration

print(f"Configuration: {layers}")

graph = {}
activations = {}

#
# Adding inputs - creating neurons
#

shared_bias_groups = []
shared_weight_groups = []

# input layer
for i in range(1,layers[0]+1):
    activations = add_input(activations,neuron_id = i,input_id = 1,act_type = 0,modifiable=False)

# hiddens
for layer_id, hidden_num in enumerate(layers[1:-1]):
    
    startind = max(list(activations.keys()))
    for neuron_id in range(hidden_num):
        
        activations = add_input(activations,neuron_id = neuron_id + startind + 1,input_id = 1,act_type = act_type,modifiable=True)
        if (layer_id + 1) == len(layers)//2 and family == "v2":
            activations = add_input(activations,neuron_id = neuron_id + startind + 1,input_id = 2,act_type = 1,modifiable=True)
            
# output layer
startind = max(list(activations.keys()))
for i in range(1,output_num+1):
    activations = add_input(activations,neuron_id = startind + i,input_id = 1,act_type = 0,modifiable=True)

#
# Adding neighbours
#

# v0 model 
ind_to = 0; ind_from = 0
for layer in range (len(layers)-1):
    ind_to=ind_from+layers[layer]
    
    for k in range(layers[layer]):
        
        neuron_id = ind_from + k + 1
        for l in range(layers[layer+1]):
            shared_group = []        
            neighbor_n_id = ind_to + l + 1
            
            graph = add_neighbor(graph,neuron_id = neuron_id, neighbor_n_id = neighbor_n_id,neighbor_i_id= 1,modifiable=True)
            
        # v2 model
        if family == "v2" and layer == len(layers)//2:
            neuron_id = ind_from + k + 1
            for l in range(layers[layer]):
                if k!=l:
                    neighbor_n_id = ind_from + l + 1
                    graph = add_neighbor(graph,neuron_id = neuron_id, neighbor_n_id = neighbor_n_id,neighbor_i_id=1,modifiable=True)
                                    
    ind_from=ind_from+layers[layer]
    
# v1 model
if family == "v1":
    ind_start = sum(layers[:len(layers)//2])
    mlayerlen = layers[len(layers)//2]

    for k in range(mlayerlen):
        for l in range(mlayerlen):
            s_id = ind_start + k + 1
            t_id = ind_start + l + 1
        
            if t_id != s_id:
                graph = add_neighbor(graph,neuron_id = s_id, neighbor_n_id = t_id,neighbor_i_id=1,modifiable=True)
   

#
# Creating the shared weights file
#
print(f"Shared weight groups: {len(shared_weight_groups)}")
to_file_shared_w = []
for group in shared_weight_groups:
    line = str(len(group))+" ### "
    for weight in group:
        line += f"{weight[0]} {weight[1]} {weight[2]}; "
    to_file_shared_w.append(line)

#
# Creating the shared bias file
#
print(f"Shared bias groups: {len(shared_bias_groups)}")
to_file_shared_b = []
for group in shared_bias_groups:
    line = str(len(group))+" ### "
    for bias in group:
        line += f"{bias[0]} {bias[1]}; "
    to_file_shared_b.append(line)

#
# Converting to list
#

to_file_graph = []
to_file_logic = []
to_file_fixwb = []
trainable_2 = 0

for line_ind in sorted(graph):
    neighbors = graph[line_ind]
    line_graph = str(len(neighbors))+ " ### "
    line_fixwb = ""
    line_logic = ""
    
    # neighbors
    for neighbor in neighbors:
        line_graph += f"{neighbor[0]} {neighbor[1]}; "
        logic_switch = int(neighbor[2])
        line_logic += f"{logic_switch} "
        
        if (neighbor[2]==False):
            line_fixwb += f"{neighbor[3]} "
            
    # activations
    line_graph += "### "
    line_logic += "### "
    line_fixwb += "### "
    activations_neuron = activations[line_ind]
    
    line_graph += str(len(activations_neuron)) + " ### "
    for activation_id in sorted(activations_neuron):
        activation = activations_neuron[activation_id]
        line_graph += f"{activation[0]} "
        
        logic_switch = int(activation[1])
        line_logic += f"{logic_switch} "
        
        if (activation[1]==False):
            line_fixwb += f"{activation[2]} "
    
    to_file_graph.append(line_graph)
    to_file_logic.append(line_logic)
    to_file_fixwb.append(line_fixwb)
    
neuron_num = len(graph)
print(f"Neuron num: {neuron_num}")

#
# Calculating the amount of the trainable parameters
#
trainable_parameters = 0
for line in to_file_logic:
    trainable_parameters += line.count('1')

for group in shared_bias_groups:
    
    neuron_id = group[0][0]
    input_id = group[0][1]
    logic = activations[neuron_id][input_id][1]
    if logic == 1:
        trainable_parameters -= len(group) - 1
        
for group in shared_weight_groups:
    

    neuron_id = group[0][0]
    neighbor_id = group[0][1]
    neighbor_input_id = group[0][2]
    
    find = 0
    i = 0
    logic = 0
    while find == 0 and i < len(graph[neuron_id]):
        neighbor_id_temp = graph[neuron_id][i][0]
        neighbor_input_id_temp = graph[neuron_id][i][1]
        logic = graph[neuron_id][i][2]
        if neighbor_id_temp == neighbor_id and neighbor_input_id_temp == neighbor_input_id:
            find = 1
        i += 1
    
    if logic == 1:
        trainable_parameters -= len(group) - 1

print(f"Trainable parameters: {trainable_parameters}")

#
# Saving the outputs ---> these are the inputs of the C code
#

graph_datas = f"graph_autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}.dat"
f = open(graph_datas,"w")
for line in to_file_graph:
    f.write(line+"\n")
f.close()

logic_datas = f"logic_autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}.dat"
f = open(logic_datas,"w")
for line in to_file_logic:
    f.write(line+"\n")
f.close()

fixwb_datas = f"fixwb_autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}.dat"
f = open(fixwb_datas,"w")
for line in to_file_fixwb:
    f.write(line+"\n")
f.close()

shared_w_datas = f"shared_w_autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}.dat"
f = open(shared_w_datas,"w")
for line in to_file_shared_w:
    f.write(line+"\n")
f.close()

shared_b_datas = f"shared_b_autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}.dat"
f = open(shared_b_datas,"w")
for line in to_file_shared_b:
    f.write(line+"\n")
f.close()

print(f"Id: *_autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}.dat")


