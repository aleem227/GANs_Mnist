# LSGANs for MNIST Digit Generation

This project implements a Least Squares Generative Adversarial Network (LSGAN) to generate MNIST handwritten digits using PyTorch.

## Overview

LSGANs are a variant of GANs that use least squares loss function instead of the traditional cross-entropy loss. This helps provide more stable training by avoiding the vanishing gradients problem that can occur with regular GANs.

## Implementation Details

### Architecture

The implementation consists of:

#### Generator
- Input: Random noise vector of size 100 (latent_dim)
- Architecture:
  - Linear layer to project and reshape
  - 3 Transposed Convolution layers with BatchNorm and ReLU
  - Output: 28x28 grayscale images

#### Discriminator  
- Input: 28x28 grayscale images
- Architecture:
  - 3 Convolution layers with BatchNorm and LeakyReLU
  - Final linear layer with sigmoid activation
  - Output: Real/Fake prediction

### Key Features

- LSGAN loss function using MSE Loss
- Data augmentation with random horizontal flips and rotations
- Batch normalization for stable training
- Added noise to generator input for improved stability
- Progress visualization every 10 epochs

### Hyperparameters

- Latent dimension: 100
- Hidden dimension: 256 
- Batch size: 32
- Learning rate: 3e-4
- Beta1: 0.5
- Number of epochs: 70

## Training

The model is trained on the MNIST dataset with:
- Alternating training of generator and discriminator
- Progress tracking with loss metrics
- Regular visualization of generated samples
- Model checkpoints saved after training

## Results

Generated samples are saved in the `saved_figure` directory every 10 epochs, showing the progression of the generator's ability to create realistic digit images. The final trained models are saved as `generator.pth` and `discriminator.pth`.

## Requirements

- PyTorch
- torchvision
- matplotlib
- tqdm

## Usage

Simply run:`python GAN.py`
