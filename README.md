**Table of contents**

- [ELSR-torch](#elsr-torch)
  * [Requirements](#requirements)
  * [Dataset](#dataset)
  * [Model](#model)
	+ [PixelShuffle](#pixelshuffle)
  * [Usage](#usage)
  * [Training](#training)
    + [Training step 1](#training-step-1)
    + [Training step 2](#training-step-2)
    + [Training step 3](#training-step-3)
  * [Results](#results)
	+ [Tests](#tests)


# ELSR-torch
Implementation of the paper ["ELSR: Extreme Low-Power Super Resolution Network For Mobile Devices"](https://arxiv.org/abs/2208.14600) using PyTorch. The code replicates the method proposed by the paper, but it is meant to be trained on limited devices. For that purpose the dataset is drastically smaller, and the training is way simpler.

## Requirements
In order to run the code on your machine you will need the packages listed on the file requirements.txt. I suggest using Anaconda, run:
```bash
conda create -n elsr --file requirements.txt 
```
Once installed the required packages, download the [dataset](https://drive.google.com/drive/folders/158bbeXr6EtCiuLI5wSh3SYRWMaWxK0Mq?usp=sharing) I used to run the training. Alternatively you can download the entire REDS dataset from [here](https://seungjunnah.github.io/Datasets/reds.html).

## Dataset
ELSR is trained on the REDS dataset, composed of sets of 300 videos each with a different degradation. This model is trained on a drastically reduced version of the dataset, containing only 30 videos with lower resolution (the original dataset was too big for me to train). The dataset (h5 files) is available at the following link: [https://drive.google.com/drive/folders/158bbeXr6EtCiuLI5wSh3SYRWMaWxK0Mq?usp=sharing](https://drive.google.com/drive/folders/158bbeXr6EtCiuLI5wSh3SYRWMaWxK0Mq?usp=sharing).
To prevent overfitting and achieve better training results, I've done some random data augmentation (see augment_data() in preprocessing.py). An example of augmentation by rotation is shown below:

![](/plots/aug.png)

## Model
The ELSR model is a small sub-pixel convolutional neural network with 6 layers, only 5 of them are learnable. The architecture is shown in the image below:

![](/plots/elsr.png)

### PixelShuffle
The PixelShuffle block (also known as depth2space) that performs computationally efficient upsampling by rearranging pixels in an image to increase its spatial resolution. Formally, let **x** be a tensor of size (**batch_size**, **C_in**, **H_in**, **W_in**), where **C_in** is the number of input channels, **H_in** and **W_in** are the height and width of the input, respectively. The goal of PixelShuffle is to upsample the spatial resolution of **x** by a factor of **r**, meaning that the output should be a tensor of size (**batch_size**, **C_out**, **H_in** * **r**, **W_in** * **r**), where **C_out** = **C_in** // **r^2**.

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
The training of the ELSR model is split in 6 steps in the paper, using different loss functions and different frame patch sizes. Nonetheless, for this implementation the images in the dataset are much smaller, hence only 3 steps are needed since we can use full-size images. Notice the number of epochs is reduced and the learning rate scheduler of the first training step is used even in the others.

### Training step 1
Train the model on the x2 dataset using the L1 loss:
```bash
python training.py \
	--train "datasets/h5/train_X2.h5" \
	--val "datasets/h5/val_X2.h5" \
	--out "checkpoints/" \
	--scale 2 \
	--epochs 300 \
	--loss "mae" \
	--lr 0.02
```

### Training step 2
Fine-tune the pre-trained model from step 1 using the x4 dataset. Use L1 loss and use a higher learning rate. In the paper this is done in 2 steps, using different patch-sizes.
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
Fine-tune the pre-trained model from step 2 using the x4 dataset. Use MSE loss and use a lower learning rate. In the paper this is done in 3 steps, using different patch-sizes.
```bash
python training.py \
	--train "datasets/h5/train_X4.h5" \
	--val "datasets/h5/val_X4.h5" \
	--out "checkpoints/" \
	--scale 4 \
	--epochs 250 \
	--loss "mse" \
	--lr 5e-3 --weights "best_X4_model.pth"
```

## Results
Due to the limited size of the dataset I wasn't able to replicate the papers results, but indeed there are interesting results proving that video-super-resolution can be done in such a small model. The graphs below are the training losses through each training step:

![](/plots/training_losses.png)

### Tests

The testing of single frame super-resolution is done in this way (video-sr is achieved by iterating sr on every frame):
 1. Resize the input image to (image.height // upscale_factor, image.width // upscale_factor) using Bicubic interpolation
 2. Calculate the bicubic_upsampled image of the previously produced low resolution image by the same upscaling factor using Bicubic interpolation
 3. Use the low resolution image to predict the sr_image
 4. Calculate PSNR between sr_image and bicubic_upsampled
The results are shown below:

![](/plots/sanremo_upscaled.png)

The PSNR of the generated image has shown to be lower, but the resulting images are smoother, making bigger images better-looking:

![](/plots/sonic_upscaled.png)

Blurring stands out in pixelated images:

![](/plots/pika_upscaled.png)