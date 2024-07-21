# Parking Lot Segmentation with U-Net

This repository contains the implementation and datasets for a parking lot segmentation project using U-Net models. The project explores the effectiveness of U-Net architectures in segmenting parking lot zones from aerial images, with and without pretraining.

## Project Structure

- `Architetture.py`: Contains the U-Net model architectures.
- `dataloader/`: Directory containing JSON files for data loading.
- `Dataset/`, `Dataset_edge_smaller/`, `Dataset_splittato/`, `image_newmask/`, `resized_masks512/`, `ResizedDataset_even_smaller/`: Directories with datasets and processed images.
- `Preprocessing.ipynb`: Jupyter notebook for data preprocessing steps.
- `Unet-noPretrain.ipynb`: U-Net model training without pretraining.
- `Unet-pretrained.ipynb`: U-Net model training with pretraining.
- `wandb/`: Weights & Biases logs and configurations.


### 1. Preprocessing.ipynb

This notebook is responsible for preparing the dataset for training. Key operations include:

- Resizing images to a uniform 512x512 dimension
- Normalizing pixel values
- Possible data augmentation (rotation, flipping, etc.)
- Splitting the dataset into training, validation, and test sets

### 2. Unet-noPretrain.ipynb

This notebook contains the code for setting up, training, and evaluating a U-Net model without using pretrained weights. Key components:

- Model: U-Net architecture initialized from scratch
- Loss Function: nn.BCEWithLogitsLoss()
- Optimizer: torch.optim.Adam with learning rate 1e-4
- Training: 30 epochs
