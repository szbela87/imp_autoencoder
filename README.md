# On the computation of the gradient in implicit neural networks

This repository contains the code for the preprint:
Béla J. Szekeres and Ferenc Izsák: On the computation of the gradient in
implicit neural networks

# Getting Started

First of all, 
the code will be able to run with an NVIDIA graphics card, for example, a GeForce GTX 1080 is required. We tested the code on Ubuntu 20.04 operating system with a Geforce GTX 1080 8GB graphics card. The driver version is 470.182.03, and the CUDA version is 11.4.

Please make sure you have installed Conda. Then in the base environment install a few necessary libraries with the following commands.
```
pip install pandas numpy seaborn scipy sklearn matplotlib xgboost hyperopt 
pip install -U imbalanced-learn
```

The source code of the CUDA C program can be found in the `imp_network` directory.
You can compile it in the following way:
```
make clean
make
```

# Documentation
A brief user manual can be found in the `doc` folder.

# Dataset
The data files can be found in the `data` directory.
They were created with the Jupyter notebook in the same place.
The resampled dataset by SMOTE algorithm can be created by `create_smote.ipynb` notebook.
The original dataset (HTRU_2.csv) can be also downloaded from:

* https://archive.ics.uci.edu/dataset/372/htru2 
* https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2


# Configurations
The autoencoder configurations can be created by `configs/create_config/conf_autoencoder.py` script.

Example:
```
python conf_autoencoder.py --hidden_layer_num 5 --start_layer 32 --latent_num 8 --input_num 8 --act_type 9 --family v0
```

The investigated models are in the `configs` folder.

# Training
The training scripts can be found in the scripts directory 
corresponding to model families and the datasets.

Copy them to the `imp_network` directory.
If you want to train the models then just simply run them.
The filenames correspond to the model names, e.g.,
`neural_v1_5_32_16` trains the `(5;32;16)-v1` model (10 times).
All the outputs will be created in the outputs directory.

# Evaluation
Copy the `eval.py` file to the appropriate output directory. Then run it in that directory according to the model parameters found there. Here are two examples for `(7;64;16)-v0` model:

```
python eval.py --family v0 --hidden_layer_num 7 --start_layer 64 --latent_num 16 --input_num 8 > results.txt
python eval.py --family v0 --hidden_layer_num 7 --start_layer 64 --latent_num 16 --input_num 8 --conf_matrix "model (7;64;16)-v0
python eval.py --hidden_layer_num 7 --start_layer 64 --latent_num 16 --input_num 117 --vmetric F1 --dataset nsl-kdd --family v1"
```
With the first example, we write the output of the evaluation to the `results.txt` file for the HTRU2 dataset. The default dataset is HTRU2. With the second call, we do not write to a file, but we also generate the confusion matrix with the title "model (7;64;16)-v0".
In the third example, `dataset` argument is given. Here, we can choose between `pulsar` and `nsl-kdd`. The first one is the default (assigned to HTRU2).

# Inference times
Copy the scripts located in the scripts_inference_times directory into the `neural_imp` folder. Run them individually. For example, you can evaluate the `(5,32,8)-v0` model using the following command:

```
./neural_v0_5_32_8_inf.sh
./neural_v1_7_64_16_nsl-kdd_inf.sh
```
The output will be created in the `outputs` folder. After that, copy the `eval.py` file into the `outputs` folder, and then run it. In the previous examples, use the following parameters:
```
python eval.py --family v0 --hidden_layer_num 5 --start_layer 32 --latent_num 8 > results.txt
python eval.py --hidden_layer_num 7 --start_layer 64 --latent_num 16 --input_num 117 --vmetric F1 --dataset nsl-kdd --family v1
```
The result will be saved in `results.txt`. You will find the final results in the `results_inference_times folder`.

# Results
For the reproducibility, all the saved results with the saved models can be found in the `results` folder in the case of the HTRU2 dataset.
On the other hand, all inference time results and best models, predictions are available at [Google Drive](https://drive.google.com/drive/folders/1aO-cc4ESeyfuXWKX0u_2KxPf-FJJhePX?usp=sharing).
