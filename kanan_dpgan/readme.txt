Differentially Private Time Series Generation with DP-GAN and TimeGAN
📌 Author
Kanan ORUJOV
🧠 Project Overview
This project aims to generate synthetic time series data while preserving individual privacy. By integrating Differentially Private Generative Adversarial Networks (DP-GAN) and modifying TimeGAN to incorporate differential privacy mechanisms, the model ensures realistic data generation without compromising sensitive information.​

🛠 Tools & Technologies
Programming Language: Python

Deep Learning Libraries: TensorFlow, PyTorch

Privacy Framework: Opacus (for implementing differential privacy)

Data Processing: Pandas, NumPy, Scikit-learn​

🔍 Methodology
Differential Privacy Mechanisms: Implemented techniques like Laplace and Gaussian noise addition, and Differentially Private Stochastic Gradient Descent (DP-SGD) to ensure data privacy.

DP-GAN: An extension of GANs that integrates differential privacy to generate realistic data while minimizing the risk of sensitive information leakage.

TimeGAN Modification: Adapted TimeGAN architecture to incorporate differential privacy, ensuring the synthetic data retains temporal dependencies.​

📁 Project Structure
generate_data.py: Script for data preprocessing and preparation.

train_dpgan.py: Training script for the Differentially Private GAN model.

train_dp_timegan.py: Training script for the modified TimeGAN with differential privacy.

📈 Results
The models successfully generate synthetic time series data that maintain the statistical properties of the original dataset while ensuring differential privacy. Evaluation metrics and visualizations can be found in the results/ directory.
