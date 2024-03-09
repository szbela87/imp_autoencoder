"""
Evaluation script

Parameters:
- family: model family - v0, v1, v2
- hidden_layer_num: number of the hidden layers
- start_layer: the size of the first layer in the encoder
- latent_num: the size of the latent layer
- input_num: the size of the input
- conf_matrix: if 1 the plot the best model confusion matrix

Example:
python eval.py --family v0 --hidden_layer_num 7 --start_layer 64 --latent_num 16 --input_num 8 > results.txt
python eval.py --family v0 --hidden_layer_num 7 --start_layer 64 --latent_num 16 --input_num 8 --conf_matrix "model (7;64;16)-v0"
"""

import pandas as pd
import numpy as np
import argparse
import os
from scipy import stats
from sklearn.metrics import matthews_corrcoef, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument('--family',type=str,default="v0")
parser.add_argument('--hidden_layer_num',type=int,default=7)
parser.add_argument('--start_layer',type=int,default=64)
parser.add_argument('--latent_num',type=int,default=8)
parser.add_argument('--input_num',type=int,default=8)
parser.add_argument('--num_sims',type=int,default=10)
parser.add_argument('--conf_matrix',type=str,default="")
parser.add_argument('--vmetric',type=str,default="AUC")

args = parser.parse_args()

family = args.family # Model family
hidden_layer_num = args.hidden_layer_num # Hidden layers
start_layer = args.start_layer # Size of the first encoder layer
latent_num = args.latent_num # Latent layer size
input_num = args.input_num # Input size - also the output size
num_sims = args.num_sims # Amount of simulations
vmetric = args.vmetric # Validation metric

ff_opt=1
if family == "v0":
    ff_opt = 1
else:
    ff_opt = 0
    
print(f"Configuration: autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}.dat")

# Reading the column names
conf_id = 1
file_path = f"test_metrics_final_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}_{conf_id}.dat"
lines = []
with open(file_path, 'r') as file:
    for line in file:
        lines.append(line.strip())
        
del lines[0]
del lines[-1]
del lines[1]

columns = []

line = lines[0]
columns = line.split("|")
columns = [column.strip() for column in columns]
columns = columns[1:-1]

data_all = []

# Loading the the data
for conf_id in range(1,num_sims+1):
    file_path = f"test_metrics_final_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}_{conf_id}.dat"
    lines = []
    with open(file_path, 'r') as file:
        
        for line in file:
            lines.append(line.strip())
        
    #for line in lines:
    #    print(line)
    del lines[2]
    del lines[1]
    del lines[0]
    line = lines[0]
    data_line = line.split("|")
    data_line = [column.strip() for column in data_line]
    data_line = data_line[1:-1]
    data_line = [float(element) for element in data_line ]
    data_all.append(data_line)
    
data_all = np.array(data_all)
df = pd.DataFrame(data_all, columns=columns)
#print(columns)
df = df.drop(["ITER"],axis=1)
print(f"\nThe results: \n------------\n{df}\n")

max_index = df[f'V{vmetric}'].idxmax()
print(f"The index: {max_index+1} (for the best V{vmetric})\n-------------")
print(f"Corresponding results to the best validation {vmetric}:\n-------------------------------------------------\n{df.loc[max_index]}\n")

results = pd.DataFrame(columns=['AVG', 'PM'])

for column in df.columns:
    mean = df[column].mean()
    ci_low, ci_up = stats.t.interval(0.95, len(df[column])-1, loc=mean, scale=stats.sem(df[column]))
    results.loc[column] = [mean, mean-ci_low]

#averages = df.mean(axis=0)
print(f"Average results:\n----------------\n{results}\n")

# Creating the confusion matrix for the best model
file_path = f"predict_test_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}_{max_index + 1}.dat"
predict_df = pd.read_csv(file_path,sep=" ",header=None)
predict_df = predict_df.to_numpy()[:,:-1]
test_df = pd.read_csv("../../data/htru2_test.csv",sep=" ",header=None).to_numpy()
X_test = test_df[:,:-1]
y_test = test_df[:,-1]
backup_path = f"backup_best_autoencoder_{family}_M{hidden_layer_num}_S{start_layer}_L{latent_num}-I{input_num}_{max_index+1}.dat"
lines = []
with open(backup_path, 'r') as file:
    # Read and print each line in the file
    for line in file:
        lines.append(line.strip())

valid_th = float(lines[-1].split(" ")[0])
valid_mean = float(lines[-1].split(" ")[1])
valid_std =  float(lines[-1].split(" ")[2])
print(f"valid_th: {valid_th} | valid_mean: {valid_mean} | valid_std {valid_std}")

errors = np.sqrt(np.mean((X_test - predict_df)**2.0,axis=1))
errors = (errors - valid_mean)/valid_std

y_pred = np.zeros_like(y_test)
y_pred[errors > valid_th] = 1

TP = ((y_pred == 1) & (y_test == 1)).sum()
FP = ((y_pred == 1) & (y_test == 0)).sum()
TN = ((y_pred == 0) & (y_test == 0)).sum()
FN = ((y_pred == 0) & (y_test == 1)).sum()

# Calculate TPR and FPR
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
                  
# Calculate MCC
numerator = TP * TN - FP * FN
denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
# Handle division by zero
mcc = numerator / denominator if denominator != 0 else 0.0

def create_confusion_matrix(TP, TN, FP, FN, title):
    # Transposed confusion matrix creation
    confusion_matrix_transposed = np.array([[TN, FN],  # Swap FN and FP to transpose
                                            [FP, TP]])
    
    # Displaying the transposed confusion matrix
    ax = sns.heatmap(confusion_matrix_transposed, annot=True, cmap='Blues', fmt='g', annot_kws={"size": 14})  # Increase annotation font size
    ax.set_title(title, fontsize=16,fontweight='bold')  # Uncomment to set title with larger font size
    ax.set_xlabel('Actual', fontsize=14,fontweight='bold')  # Increase font size for x-axis label
    ax.set_ylabel('Predicted', fontsize=14,fontweight='bold')  # Increase font size for y-axis label
    ax.xaxis.set_ticklabels(['Non-pulsar', 'Pulsar'], fontsize=14, rotation=0)  # Increase font size for x-axis tick labels and rotate
    ax.yaxis.set_ticklabels(['Non-pulsar', 'Pulsar'], fontsize=14, rotation=0)  # Increase font size for y-axis tick labels and rotate
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.savefig('confusion_matrix.png', dpi=300)  # Save the figure to a file
    plt.show()

if args.conf_matrix!="":
    create_confusion_matrix(TP,TN,FP,FN,args.conf_matrix) 

