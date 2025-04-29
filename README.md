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

Perfect!  
You want a GitHub-style, full, professional README — **similar to the examples you sent** — but matching **your project**:  
(**Synthetic Time Series Generation using TimeGAN and Model Inversion, done by you, Royal Abdullazade**).

Here’s a complete README you can use directly for your GitHub or project submission:

---

# **Synthetic Time Series Generation using TimeGAN and Model Inversion**

## 📋 Project Description

This project implements a TimeGAN-inspired approach to generate synthetic time series data, with an exploration of model inversion techniques to improve the realism and fidelity of generated sequences.  
The work was completed as part of a team project — this part was developed by **Royal Abdullazade**.

The main goal was to create synthetic sequential datasets that preserve temporal dependencies while enhancing data fidelity through early model inversion ideas.  
The project covers data preprocessing, model building, training, and visualization.

---

# 🔁 Repeatable Experiment: Step-by-Step Instructions

## Environment Setup

Install required libraries:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```
Make sure you are using **Python 3.8+** and **TensorFlow 2.x**.

---

## Generate Synthetic Data
- Generate a synthetic sequential dataset using random functions.
- Normalize the dataset using `MinMaxScaler` to [0,1] range for stable training.

---

## Build the TimeGAN-Inspired Model
- Define the **Generator** using LSTM layers.
- Define the **Discriminator** using LSTM layers.
- Implement a simple adversarial training loop.
- Adapt TimeGAN principles to maintain sequential (time-dependent) properties.

---

## Training the Model
- Train both generator and discriminator on normalized data.
- Losses include:
  - **Generator loss**: Encourage realistic synthetic sequences.
  - **Discriminator loss**: Distinguish between real and synthetic sequences.
- Basic model inversion idea: attempt to reconstruct meaningful inputs from synthetic outputs.

---

## Visualize Results
- Plot real time series data.
- Plot generated synthetic time series.
- Compare the shapes, fluctuations, and patterns between real and synthetic sequences using Matplotlib.

---

# ⚙️ Technologies Used

- **Python 3.8+**
- **TensorFlow 2.x**
- **NumPy**
- **Matplotlib**
- **scikit-learn**

---

# 📈 Project Workflow

1. **Dataset Creation**  
Random time series data generation.
   
2. **Data Preprocessing**  
Normalization using `MinMaxScaler`.

3. **Model Building**  
Simple LSTM-based Generator and Discriminator.

4. **Training**  
Model training on synthetic time series data.

5. **Model Inversion Exploration**  
Testing early ideas on reversing synthetic outputs to input space.

6. **Evaluation**  
Plotting and analyzing the synthetic vs real data.

---

# 👤 Author

- **Royal Abdullazade** — Part of a team project; contributed the TimeGAN-based synthetic data generation and initial model inversion concept development.

---

# 📚 References

1. Yoon, Jinsung, Daniel Jarrett, and Mihaela van der Schaar (2019). "Time-series Generative Adversarial Networks." *Advances in Neural Information Processing Systems (NeurIPS)*. [Link](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks)
   
2. Goodfellow, Ian, et al. (2014). "Generative Adversarial Networks." *Communications of the ACM*, Vol. 63 No. 11, Pages 139-144. [Link](https://arxiv.org/abs/1406.2661)
   
3. Chollet, François. (2015). *Keras: Deep Learning for Humans*. GitHub Repository. [Link](https://github.com/keras-team/keras)
   
4. Scikit-learn Developers. (2024). "scikit-learn: Machine Learning in Python." [Link](https://scikit-learn.org/)
   
5. TensorFlow Developers. (2024). "TensorFlow: An End-to-End Open Source Machine Learning Platform." [Link](https://www.tensorflow.org/)

