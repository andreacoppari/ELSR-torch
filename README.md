**Table of Contents**

[TOC]

# ELSR-torch
Implementation of the paper "ELSR: Extreme Low-Power Super Resolution Network For Mobile Devices" using PyTorch. The code replicates the method proposed by the paper, but it is meant to be trained on limited device. For that purpose the dataset is drastically smaller, and the training is way simpler.

## Prerequisites
In order to run the code on your machine you will need python>=3.9 and the packages listed on the file requirements.txt. I suggest using Anaconda, run:
```bash
conda create -n elsr --file requirements.txt python -c conda-forge
```
Once installed the required packages, download the [dataset]() I used to run the training. Alternatively you can download the entire REDS dataset from [here](https://seungjunnah.github.io/Datasets/reds.html)

## Dataset
ELSR is trained on the REDS dataset, composed of sets of 300 videos each with a different degradation. This model is trained on a drastically reduced version of the dataset, containing only 30 videos with lower resolution (the original dataset was too big for me to train). The dataset (h5 files) is available at the following link: []()

## Usage
To train the model run:
```bash
python training.py
	--train <training_dataset_path>
	--val <validation_dataset_path>
	--out <path_for_best_model>
	--weights <weights_path(not required)>
```
To test the model run:
```bash
python training.py
	--weights <weights_path(not required)>
	--input <input_frames_path>
```

## Results
