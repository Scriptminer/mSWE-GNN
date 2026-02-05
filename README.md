# mSWE-GNN (Repository for paper "Multi-scale hydraulic graph neural networks for flood modelling")
(Version 1.1 - Nov. 28th, 2024)

![Architecture](Architecture.png)

## Overview

For reproducing the paper's results, explore **plot_results.ipynb**

For training the model run **main.py**

For training and exploring the model, run **main.ipynb**

For testing a model, run **test_model.py**

Both **main.py** and **main.ipynb** use a **config.yaml** as reference configuration file.

The repository is divided in the following folders:

* **database:** Creation of hydrodynamic simulations (**D-Hydro simulations.ipynb**) [requires the license and installation of "D-HYDRO Suite 1D2D"] and conversion of the NETCDF output files into PyTorch Geometric-friendly data (**create_dataset.ipynb**).
Also contains the output of the hydrodynamic simulations (**raw_datasets**: for downloading the datasets go to <https://doi.org/10.5281/zenodo.13326595>). This is converted into Pickle files that are then stored and separated into training and testing datasets in **datasets**.

* **models:**  Deep learning models developed for surrogating the hydraulic one: contains a base class with common inputs and functions and one for the SWE-GNN and mSWE-GNN models.

* **results:** Contains results and trained models of the mSWE-GNN Pareto front, used for the paper's results.

* **training:** Contains loss and training functions.

* **utils:** Contains Python functions for loading, creating and scaling the dataset. There are also other miscellaneous functions and visualization functions.

## Environment setup

The required libraries are in requirements.txt.
Requires Python 3.10.
