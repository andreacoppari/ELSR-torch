**Table of contents**

- [ELSR-torch](#elsr-torch)
  * [Prerequisites](#prerequisites)
  * [Dataset](#dataset)
  * [Usage](#usage)
  * [Results](#results)

# ELSR-torch
Implementation of the paper ["ELSR: Extreme Low-Power Super Resolution Network For Mobile Devices"](https://arxiv.org/abs/2208.14600) using PyTorch. The code replicates the method proposed by the paper, but it is meant to be trained on limited device. For that purpose the dataset is drastically smaller, and the training is way simpler.

## Prerequisites
In order to run the code on your machine you will need the packages listed on the file requirements.txt. I suggest using Anaconda, run:
```bash
conda create -n elsr --file requirements.txt python -c conda-forge -c pytorch -c nvidia
```
Once installed the required packages, download the [dataset](https://drive.google.com/drive/folders/158bbeXr6EtCiuLI5wSh3SYRWMaWxK0Mq?usp=sharing) I used to run the training. Alternatively you can download the entire REDS dataset from [here](https://seungjunnah.github.io/Datasets/reds.html).

## Dataset
ELSR is trained on the REDS dataset, composed of sets of 300 videos each with a different degradation. This model is trained on a drastically reduced version of the dataset, containing only 30 videos with lower resolution (the original dataset was too big for me to train). The dataset (h5 files) is available at the following link: [https://drive.google.com/drive/folders/158bbeXr6EtCiuLI5wSh3SYRWMaWxK0Mq?usp=sharing](https://drive.google.com/drive/folders/158bbeXr6EtCiuLI5wSh3SYRWMaWxK0Mq?usp=sharing).

## Usage
To train the model run:
```bash
python training.py                      \
	--train <training_dataset_path>     \
	--val <validation_dataset_path>     \
	--out <path_for_best_model>         \
	--weights <weights_path(not required)>
```
To test the model run:
```bash
python training.py                          \
	--weights <weights_path(not required)>  \
	--input <input_frames_path>
```

## Training
The training of the ELSR model is split in 6 steps in the paper, using different loss functions and different frame patch sizes. Nonetheless, for this implementation the images in the dataset are much smaller, hence only 3 steps are needed since we can use full-size images.

### Training step 1
```bash
python training.py \
	--train "datasets/h5/train_X2.h5" \
	--val "datasets/h5/val_X2.h5" \
	--out "checkpoints/" \
	--scale 2 \
	--epochs 250 \
	--loss "mae" \
	--lr 0.02
```

### Training step 2
```bash
python training.py \
	--train "datasets/h5/train_X4.h5" \
	--val "datasets/h5/val_X4.h5" \
	--out "checkpoints/" \
	--scale 4 \
	--epochs 250 \
	--loss "mae" \
	--lr 0.05 \
	--weights "best_X2_model.pth"
```

### Training step 3
```bash
python training.py \
	--train "datasets/h5/train_X4.h5" \
	--val "datasets/h5/val_X4.h5" \
	--out "checkpoints/" \
	--scale 4 \
	--epochs 250 \
	--loss "mse" \
	--lr 0.02 --weights "best_X4_model.pth"
```

## Results
See 'plots' folder