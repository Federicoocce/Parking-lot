# Parking Lot Segmentation with U-Net

This repository contains the implementation and datasets for a parking lot segmentation project using U-Net models. The project explores the effectiveness of U-Net architectures in segmenting parking lot zones from aerial images, with and without pretraining.

## Project Structure

- `Architetture.py`: Contains the U-Net model architectures.
- `dataloader/`: Directory containing JSON files for data loading.
- `intermediate_datasets/`: Directory with datasets and processed images used during the process.
- `dataset_divided_inclasses/`: Directory with the dataset divided in the 3 quality classes
- `Preprocessing.ipynb`: Jupyter notebook for data preprocessing steps.
- `Unet-noPretrain.ipynb`: U-Net model training without pretraining.
- `Unet-pretrained.ipynb`: U-Net model training with pretraining.
- `wandb/`: Weights & Biases logs and configurations.
- `livelli_park.txt`: List of the classes assigned for each image.


### 1. Preprocessing.ipynb

This notebook is responsible for preparing the dataset for training. Key operations include:

- Resizing images to a uniform 512x512 dimension
- Dividing the images of the dataset in 3 classes
- Application of data augmentation techniques(rotation, flipping, etc.)
- Splitting the dataset into training, validation, and test sets

- There are also portions of code that were used in previous attempts that involved padding

### 2. Unet-noPretrain.ipynb

This notebook contains the code for setting up, training, and evaluating a U-Net model without using pretrained weights. Key components:

- Model: U-Net architecture initialized from scratch
- Loss Function: nn.BCEWithLogitsLoss() combined with dice loss
- Optimizer: torch.optim.Adam with learning rate 1e-3
- Training: 50 epochs

### 3. Unet-pretrain.ipynb
This notebook contains the code for setting up, training, and evaluating a pre-trained U-Net model. Key components:
- Model: U-Net architecture with pre-trained weights
- Loss Function: nn.BCEWithLogitsLoss() combined with dice loss
- Optimizer: torch.optim.Adam with learning rate 1e-3
- Training: 30 epochs
## How to run the code
Augmented dataset are splitted in 6 folder: train_images Test_images and test_images for the input, same folder pattern for the output masks folders.
Code was mostly run on kaggle so the folders path must be changed.
After the folder path changes is sufficient to run the Unet-nopretrain notebook to run the training and testing of the custom Unet or Unet-pretrainet for the pretrained counterpart.
## Authors
Barbieri Lorenzo, Occelli Federico
