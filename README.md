# Time Series Synthetic Data Generation via Formal Methods

## Overview

This project proposes a novel approach to synthetic time series data generation using **formal methods** instead of traditional deep learning models.  
We model time-dependent behaviors using **timed automata** and generate synthetic sequences through simulation with **UPPAAL**.  
This method provides a reliable, scalable, and interpretable alternative to GANs and RNN-based approaches.

## Motivation

- Many domains (healthcare, finance, engineering) suffer from limited or sensitive datasets.
- Traditional deep learning models struggle with long-term dependencies, training instability, and mode collapse.
- Formal methods offer control, interpretability, and stability for synthetic data generation.

## Methodology

- **Modeling**: Build timed automata to capture the temporal dynamics of the system.
- **Simulation**: Use UPPAAL to simulate and generate synthetic time series sequences.
- **Data Extraction**: Convert simulation outputs into structured datasets.

## Why Formal Methods?

- **Explainability**: Models are interpretable and transparent.
- **Temporal Coherence**: Naturally encode long-range dependencies.
- **Stability**: Avoid adversarial training instability.
- **Privacy**: No direct exposure of real sensitive data.

## Tools

- [UPPAAL](https://uppaal.org/) — for modeling and simulation
- Python scripts (optional) — for automating simulation runs and parsing outputs



# **Variational Autoencoder for Synthetic Timeseries Data**
# 📋 Project Description :

This project implements a Variational Autoencoder (VAE) model to learn and reconstruct synthetic timeseries data. The model is built using TensorFlow (Keras API) and focuses on encoding high-dimensional timeseries into a lower-dimensional latent space, and then decoding them back to approximate the original data.
The goal is to visualize how well the VAE can learn patterns from the timeseries and generate similar sequences. The project also includes custom training, visualization of the original and reconstructed timeseries, and dimensionality reduction techniques.


# 🔁 Repeatable Experiment: Step-by-Step Instructions

## Environment Setup

Install required libraries:

```bash
pip install numpy matplotlib seaborn tensorflow scikit-learn
```
Ensure you are using Python 3.7+ and TensorFlow 2.x.

## Generate Synthetic Data
-Use the generate_timeseries_data() function to create random timeseries data.<br>
-Normalize the data (zero mean, unit variance).

## Build the Variational Autoencoder (VAE)
Define the encoder with two outputs: latent mean and log-variance.<br>
Define the decoder to reconstruct data from the latent space.

## Training the Model
Train the VAE with a custom loop using tf.GradientTape.<br>
The loss function combines:<br>
  Reconstruction Loss (Mean Squared Error)<br>
  KL Divergence Loss (regularizes the latent space)

## Visualize Results
Plot original synthetic timeseries.<br>
Plot reconstructed timeseries after encoding/decoding.<br>
Use Seaborn and Matplotlib for better plot aesthetics.

## Run the Code:
Adjust hyperparameters in the main() function if needed:<br>
```bash
num_samples = 3
num_timesteps = 50
latent_dim = 2
epochs = 100
batch_size = 32
```
Run the script to visualize the VAE performance.

# Technologies Used :
Python 3.7+<br>
TensorFlow 2.x (Keras API)<br>
NumPy<br>
Matplotlib<br>
Seaborn<br>
Scikit-learn (for TSNE import, although not used heavily)

# References : 

Tai, B.-C., Li, S.-C., Huang, Y., & Wang, P.-C. (2024). Examining the utility of differentially private synthetic data generated using variational autoencoder with TensorFlow Privacy.<br>

Zhang, Y., Ma, T., Li, T., Sun, X., & Liu, Z. (2024). Small sample data augmentation method for photovoltaic power generation based on improved variational auto-encoder.<br>

Liu, S., Wang, P., Chen, X., Jiang, P., Li, L., & Yin, S. (2025). An anomaly detection method for provincial-side base operation logs based on VAE and Transformer. State Grid Shandong Electric Power Company.

Liu, S., Wang, P., Chen, X., Jiang, P., Li, L., & Yin, S. (2025). An anomaly detection method for provincial-side base operation logs based on VAE and Transformer. State Grid Shandong Electric Power Company.
