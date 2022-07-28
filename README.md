[![DOI](https://zenodo.org/badge/380095338.svg)](https://zenodo.org/badge/latestdoi/380095338)

# UNSL at eRisk 2021

Repository accompanying the CLEF 2021 eRisk Workshop Working Notes for the UNSL team (Universidad Nacional de San Luis).
This repository contains the code necessary to:
- create the conda environment to run the code;
- create training and testing datasets based on Reddit for the tasks T1 and T2;
- pre-process datasets;
- train and test the proposed models (EarlyModel, EARLIEST and SS3);
- evaluate their performance using the measures ERDE and F Latency.

## Create conda environment
Make sure you have `conda` installed. If that's not the case, you can install it following this [guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
Then, to create the conda environment necessary to run the code, run:
```bash
conda env create -f environment.yml

# Activate the environment using:
conda activate unsl_erisk_2021
```

## Create datasets
To generate the datasets for T1, for example, run:
```bash
python -m datasets.make_reddit_corpus t1 keep
```
The resulting raw datasets will be stored in `datasets/interim/t1/`: `t1-train-raw.txt` for the training dataset, and `t1-test-raw.txt` for the testing dataset.

## Pre-process datasets
To pre-process the datasets for T1, run:
```bash
python -m datasets.clean_corpus t1
```
The resulting pre-processed datasets will be stored in `datasets/interim/t1/`: `t1-train-clean.txt` for the training dataset, and `t1-test-clean.txt` for the testing dataset.

## Train and test models
To train and test the models on the datasets, use the scripts under the folder `examples/`.

### EarlyModel
To evaluate the EarlyModel, run:
```bash
python -m examples.earlymodel_test t1 -n 30
```

### EARLIEST
To evaluate the EARLIEST, run:
```bash
python -m examples.earliest_test t1 cpu -n 30
```

### SS3
To evaluate the SS3, run:
```bash
python -m examples.ss3_test t1 -n 30
```

## Citation
If you use this code in a scientific publication, we would appreciate citations to the following paper:

> J. M. Loyola, S. Burdisso, H. Thompson, L. Cagnina, M. Errecalde, UNSL at eRisk 2021: A comparison of three early alert policies for early risk detection,  in: Working Notes of CLEF 2021 - Conference and Labs of the Evaluation Forum, Bucarest, Romania, September 21-24, 2021.
