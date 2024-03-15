"""
Generating the simulparams.txt files

Parameters:
- id: simulation id
- family: model family - v0, v1 or v2
- hidden_layer_num: number of the hidden layers
- start_layer: the size of the first layer in the encoder
- latent_num: the size of the latent layer
- input_num: the size of the input
- neuron_num: number of the neurons
- shared_weight_num: number of the shared weight groups
- load: no training just evaluating, if 1 then also give the `directory`
- directory: config directory
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, default=1)
parser.add_argument('--family',type=str,default="v0")
parser.add_argument('--hidden_layer_num',type=int,default=7)
parser.add_argument('--start_layer',type=int,default=64)
parser.add_argument('--latent_num',type=int,default=8)
parser.add_argument('--input_num',type=int,default=8)
parser.add_argument('--neuron_num',type=int,default=248)
parser.add_argument('--shared_weight_num',type=int,default=28)
parser.add_argument('--load',type=int,default=0)
parser.add_argument('--epochs',type=int,default=20)
parser.add_argument('--minibatch_size',type=int,default=16)
parser.add_argument('--directory',type=str,default="")
parser.add_argument('--dataset',type=str,default="pulsar_nosmote")


args = parser.parse_args()

seed = args.id + 2024 # Random seed
family = args.family # Feed forward optimization
hidden_layer_num = args.hidden_layer_num # Hidden layers
start_layer = args.start_layer # Size of the first encoder layer
latent_num = args.latent_num # Latent layer size
input_num = args.input_num # Input size - also the output size
neuron_num = args.neuron_num # Neurons' number
shared_weight_num = args.shared_weight_num # Number of the shared groups
epochs=args.epochs # epochs
minibatch_size=args.minibatch_size # mini-batch size
dataset = args.dataset # dataset name

if args.load == 1: # just evaluation
    epochs = 0
    
# Creating the inputs directory
if not os.path.exists('inputs'):
    os.makedirs('inputs')

# Creating the results directory
if not os.path.exists('outputs'):
    os.makedirs('outputs')

ff_opt = 1
if family == "v0":
    ff_opt = 1
else:
    ff_opt = 0
    
if dataset == "pulsar_nosmote":
    train_fname = '../data/htru2_train.csv'
    valid_fname = '../data/htru2_valid.csv'
    test_fname = '../data/htru2_test.csv'
    input_num = 8
    learn_num = 13014
    valid_num = 1790
    test_num = 1790
    minibatch_size = 16
    epochs = 20
    valid_metric_type = "F1"
    grad_alpha = 0.01
    alpha=0.0001
    
if dataset == "pulsar_smote":
    train_fname = '../data/htru2smote_train.csv'
    valid_fname = '../data/htru2smote_valid.csv'
    test_fname = '../data/htru2smote_test.csv'
    input_num = 8
    learn_num = 13164
    valid_num = 2929
    test_num = 1790
    minibatch_size = 16
    epochs = 20
    valid_metric_type = "F1"
    grad_alpha = 0.01
    alpha=0.0001
    
if dataset == "nsl-kdd":
    train_fname = '../data/nsl-kdd_train.csv'
    valid_fname = '../data/nsl-kdd_valid.csv'
    test_fname = '../data/nsl-kdd_test.csv'
    input_num = 117
    learn_num = 50528
    valid_num = 31494
    test_num = 22544
    minibatch_size = 32
    epochs = 5
    valid_metric_type = "F1"
    grad_alpha = 0.01
    alpha=0.0001
    #epochs = 5
    #valid_metric_type = "F1"
    #grad_alpha = 0.001
    #alpha=0.001
    
print(f"Configuration: autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}.dat")

simulparams=[
f"seed                       {seed}",
f"shuffle_num                100000",
f"thread_num                 4",
f"tol_fixit                  1.0e-5",
f"maxiter_grad               {epochs}",
f"maxiter_fix                50",
f"initdx                     1.0",
f"sfreq                      11",
f"input_name                 {train_fname}",
f"input_name_valid           {valid_fname}",
f"input_name_test            {test_fname}",
f"output_name                ./outputs/results_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}_{args.id}.dat",
f"predict_name_valid         ./outputs/predict_valid_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}_{args.id}.dat", 
f"predict_name_test          ./outputs/predict_test_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}_{args.id}.dat", 
f"test_log                   ./outputs/test_metrics_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}_{args.id}.dat", 
f"test_log_final             ./outputs/test_metrics_final_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}_{args.id}.dat", 
f"learn_num                  {learn_num}",
f"valid_num                  {valid_num}",
f"test_num                   {test_num}",
f"mini_batch_size            {minibatch_size}",
f"",
f"neuron_num                 {neuron_num}",
f"input_num                  {input_num}",
f"output_num                 {input_num}",
f"shared_weights_num         {shared_weight_num}",
f"shared_biases_num          0",
f"graph_datas                ../configs/{family}/graph_autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}.dat",
f"logic_datas                ../configs/{family}/logic_autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}.dat",
f"fixwb_datas                ../configs/{family}/fixwb_autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}.dat",
f"shared_w_datas             ../configs/{family}/shared_w_autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}.dat",
f"shared_b_datas             ../configs/{family}/shared_b_autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}.dat",
f"",
f"alpha                      {alpha}",
f"",
f"train_lossfunction_type    MSE",
f"valid_metric_type          {valid_metric_type}",
f"valid_metric_type_2        {valid_metric_type}",
f"range_div                  8000",
f"",
f"optimizer                  1",
f"nesterov                   1",
f"grad_alpha                 {grad_alpha}",
f"adam_alpha                 1.0e-3",
f"adam_beta1                 0.9",
f"adam_beta2                 0.999",
f"adam_eps                   1.0e-8",
f"",
f"lr_scheduler               2",
f"cyclic_momentum            0",
f"pcr                        1.0",
f"base_momentum              0.85",
f"max_momentum               0.95",
f"div_factor                 10.0",
f"final_div_factor           10.0",
f"step_size                  1",
f"lr_gamma                   0.5",
f"early_stopping             0",
f"",
f"ff_optimization            {ff_opt}",
f"",
f"clipping                   0",
f"clipping_treshold          1.0",
f"",
f"loaddatas                  {args.load}",
f"load_backup                ./{args.directory}backup_best_autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}_{args.id}.dat",
f"save_best_model            ./outputs/backup_best_autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}_{args.id}.dat"
]

if not os.path.exists('inputs'):
    os.makedirs('inputs')
    
if not os.path.exists('outputs'):
    os.makedirs('outputs')

fname = "./inputs/simulparams.dat"
f = open(fname,"w")
for line in simulparams:
    f.write(line+"\n")
f.close()
