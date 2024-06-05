# Code for Dynamic Spectral Clustering with Provable Approximation Guarantee
This repository provides the code to reproduce the experimental results in our paper

## Installing Dependencies
Because we use the `graph-tool` package, we highly recommend using `conda` to install dependencies for ease of installation. To install the dependencies using conda, run
```
conda env create -f environment.yml
```

and run
```
conda activate dynamic_SC
```

to activate the environment.

## Running Experiments.
To reproduce the results in Figure 2 of the paper, run
```
python sbm_experiments.py
```
This will run 10 experiments of both the increasing and decreasing $k$ scenarios, and all 20 results are saved in the `./results/` folder.

To reproduce the results in Figure 3 of the paper, run
```
python knn_experiments.py
```
This will run 10 experiments for both `MNIST` and `EMNIST`, and all 20 results are saved in the `./results/` folder.

## Plotting Results
We also provide code to reproduce the plots of Figure 2 and Figure 3 in files `./plots/plot_sbm_all.py` and `./plots/plot_mnist_emnist.py` respectively.
Run these after the experiments above have completed.
