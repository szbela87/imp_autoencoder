# On the computation of the gradient in
implicit neural networks

This repository contains the code for the paper:
Béla J. Szekeres and Ferenc Izsák: On the computation of the gradient in
implicit neural networks

## Getting Started
### Environment Requirements

First, please make sure you have installed Conda. Then in the base environment install a few libraries
```
pip install pandas numpy seaborn
```

The source code of the program can be found in the `imp_network` directory.
You can compile it in the following way:
```
make clean
make
```

# Dataset
The data files can be found in the `data` directory.
They were created with the Jupyter notebook in the same place.

# Training
The training scripts can be found in the scripts directory 
corresponding to model families.

Copy them to the `imp_network` directory.
If you want to train the models then just simply run them.
The filenames corresponds to the model names, e.g.,
`neural_v1_5_32_16` trains the (5;32;16)-v1 model (10 times).
All the outputs will be created in the outputs directory.

# Results
The saved results with the saved models can be found in the results directory.
