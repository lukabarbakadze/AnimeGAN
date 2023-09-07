# AnimeGANv2

The repository contains implementation of AnimeGANv2 using PyTorch, complete with both generator and discriminator networks. There are also a scripts which could be used for (using PyTorch Lightning) for training these networks, although this part of the project is a work in progress and may require further adjustments.

## Table of Contents
- [File Descriptions](#file-descriptions)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Acknowledgements](#acknowledgements)

## File Descriptions
- **`AnimeGAN/`**: The main directory containing the project code and resources.
  - **`model/generator.py`**: Implementation of the AnimeGANv2 generator network.
  - **`model/discriminator.py`**: Implementation of the AnimeGANv2 discriminator network.
  - **`model/AnimeGANv2.py`**: PyTorchLigntning implementation of AnimeGANv2 (not finished)
  - **`dataset/dataset.py`**: The script for building custom PyTorch DataLoader for training.
  - **`scripts/create_dataset.py`**: Script to extract images (captures an image at each scene transition.) from .mp4 videos located in /videos directory.
  - **`scripts/image_to_grayscale.py`**: Script to convert images to grayscale.
  - **`config/config.py`**: configuration file for training
  - **`train.py`**: The script for training the AnimeGANv2 model using PyTorch Lightning.
  - **`requirements.txt`**: Lists the Python dependencies required for this project.
  - **`images/`**: The directory where dataset images should be organized.
  - **`videos/`**: The directory where videos (in .mp4) should be located.

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/lukabarbakadze/AnimeGAN.git
   ```

2. Create and activate a virtual environment (e.g., using conda):
   ```sh
   conda create -n myenv python==3.9
   conda activate myenv
   ```

3. Navigate to the AnimeGAN directory and install dependencies:
   ```sh
   cd AnimeGAN
   pip install -r requirements.txt
   ```

## Dataset Preparation
Arrange your dataset as follows:
   ```sh
   AnimeGAN/
   ├── images/
   │   ├── GrayScale/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   ├── Input/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   ├── Target/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   ```

## Acknowledgements
* [Paper: AnimeGANv2](https://tachibanayoshino.github.io/AnimeGANv2/)
* [AnimeGANv2 github](https://github.com/TachibanaYoshino/AnimeGANv2)
* [PyTorch Implementation of AnimeGAN by ptran1203](https://github.com/ptran1203/pytorch-animeGAN)