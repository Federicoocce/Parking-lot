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

