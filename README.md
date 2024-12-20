# General Description
This project presents the software code for applying Kolmogorov-Arnold networks (KAN) to the inverse problem of geophysics.

Statement of the inverse problem: to determine layer boundaries using different type of sensing channels.

---
# Purposes` description of the files in this project
Directory hierarchy: 

0) Supportive code:
    * tools.py - 
        1. functions for working with JSON files; 
        2. functions for compressing input spectra;
        3. definition of class KAN_es(KAN) - KAN with early stopping based on the validation set.
    * algos.py - Functions for loading data, fitting and validating models with saving results:
        1. KAN - based on KAN_es from tools.py;
        2. mlp from keras;
        3. Random forest and Gradient boosting from skl.

1) Code demonstrating the operation of several stages of the project:
    * Overview.ipynb - Demonstrates data loading, scaling, fitting mlp and KAN with visualisations.

2) Providing pool of experiments.
    * Multi_exp_1.ipynb
    * Multi_exp_2.ipynb
    * Multi_exp_3.ipynb
    * Multi_exp_4.ipynb
    * post_process.ipynb - Providing and plotting comparative analysis of computed experiments.
---