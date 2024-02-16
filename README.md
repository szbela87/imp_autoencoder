# On the computation of the gradient in implicit neural networks

This repository contains the code for the preprint:
Béla J. Szekeres and Ferenc Izsák: On the computation of the gradient in
implicit neural networks

## Getting Started
### Environment Requirements

First of all, 
the code will be able to run with an NVIDIA graphics card, for example, a GeForce GTX 1080 is required. We tested the code on Ubuntu 20.04 operating system with a Quadro M6000 24GB graphics card. The driver version is 470.182.03, and the CUDA version is 11.4.

Please make sure you have installed Conda. Then in the base environment install a few necessary libraries with the following command.
```
pip install pandas numpy seaborn
```

The source code of the CUDA C program can be found in the `imp_network` directory.
You can compile it in the following way:
```
make clean
make
```

# Dataset
The data files can be found in the `data` directory.
They were created with the Jupyter notebook in the same place.

# Configurations
The autoencoder configurations can be created by `configs/create_config/conf_autoencoder.py` script.

Example:
```
python conf_autoencoder.py --hidden_layer_num 5 --start_layer 32 --latent_num 8 --input_num 8 --act_type 9 --family v0
```

The investigated models are in the `configs` directory.

# Training
The training scripts can be found in the scripts directory 
corresponding to model families.

Copy them to the `imp_network` directory.
If you want to train the models then just simply run them.
The filenames correspond to the model names, e.g.,
`neural_v1_5_32_16` trains the (5;32;16)-v1 model (10 times).
All the outputs will be created in the outputs directory.

# Evaluation
Copy the eval.py file to the appropriate output directory. Then run it in that directory according to the model parameters found there. Here are two examples:

```
python eval.py --family v0 --hidden_layer_num 7 --start_layer 64 --latent_num 16 --input_num 8 > results.txt
python eval.py --family v0 --hidden_layer_num 7 --start_layer 64 --latent_num 16 --input_num 8 --conf_matrix 1
```
With the first example, we write the output of the evaluation to the results.txt file. With the second call, we do not write to a file, but we also generate the confusion matrix.

# Results
For the reproducibility, all the saved results with the saved models can be found in the results directory.
