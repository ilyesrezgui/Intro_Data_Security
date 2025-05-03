# **Formal methods for time series synthetic data generation*

## 📋 Project Description

This project implements utlizes formal methods, specifcally model checking in order to generate time series data. — this part was developed by **Ilyes REZGUI**.

---

# 🔁 Repeatable Experiment: Step-by-Step Instructions

## Environment Setup

1. **Install Python 3**  
   Ensure that Python 3 is installed on your system to run the notebook.

2. **Install UPPAAL**  
   Download and install UPPAAL from [here](https://uppaal.org/) to build and visualize the model.

3. **Running the Experiment**  
   You can either:

   - **Option 1: Run the Notebook Directly**  
     Simply run the notebook to observe the data extraction process.

   - **Option 2: Start from Scratch**  
     1. Open UPPAAL and load the XML file to regenerate the synthetic data using the simulator and properties.
     2. After generating the data, run the notebook to parse and analyze it.

## Requirements

- Python 3.x
- UPPAAL installed and configured
---

# ⚙️ Technologies Used

- **Python 3.8+**
- **plotly**
- **NumPy**
- **Matplotlib**
- **scikit-learn**
- **UPPPPAL**
- **Simulate**

---

# 📈 Project Workflow

1. **Automata-Based System Representation**  
The initial step in our proposed synthetic data generation approach involves formally abstracting the system's behavior using automata.
   
2. **Formal Modeling in UPPAAL**  
After abstracting the system components using automata, we encode and implement these models in UPPAAL. UPPAAL extends classical automata with real-valued clocks, synchronization channels, guards, invariants, and user-defined variables, making it well-suited for modeling the timing constraints and complex interactions found in real-world systems.

3. **Model Verification and Data Integrity Assurance**  
Verifying the correctness of the underlying model is essential to ensure the reliability and validity of the generated data. This verification is conducted using UPPAAL, where temporal properties are defined to confirm the model's expected behavior. These properties, expressed in temporal logics like Computation Tree Logic (CTL) and simulation-based specifications, ensure that the system consistently meets the required conditions.

4. **Data Generation using UPPAAL Concrete Simulator**  
"Once the system behavior is formally modeled using UPPAAL’s timed automata, the next crucial step in our approach is leveraging the UPPAAL Concrete Simulator to generate synthetic data. This tool enables the execution of concrete instances of the timed automata, faithfully respecting timing constraints, variables, and communication protocols defined in the model. It provides a robust platform for simulating real-time system behavior and exploring diverse execution traces that reflect the dynamics of the modeled system.

6. **Data Formatting**  
"The ‘.uctr’ file captures detailed execution traces, including state transitions, variable values, and precise timing information. To enable downstream applications such as machine learning, analysis, or validation, these traces must be parsed and structured appropriately. We process the ‘.uctr’ file line by line, extracting key elements such as timestamps, transitions, and variable states. The extracted data is then organized into a structured format, such as a table or matrix. In UPPAAL models, variables represent the system’s dynamic state, encompassing both discrete state variables and real-time clock variables. These variables, logged at each transition, are crucial for accurately interpreting the system’s behavior.

---

# 👤 Author

- **Ilyes REZGUI** — Part of a team project; focused on the use of fomal methods for the generation which never was explored before in the literature.

---

# 📚 References
Jensen, P.G., Jørgensen, K.Y., Larsen, K.G., Mikučionis, M., Muñiz,
M., Poulsen, D.B., 2020. Fluid model-checking in uppaal for covid-
19, in: Leveraging Applications of Formal Methods, Verification and
Validation: Verification Principles: 9th International Symposium on
Leveraging Applications of Formal Methods, ISoLA 2020, Rhodes,
Greece, October 20–30, 2020, Proceedings, Part I 9, Springer. pp.
385–403

Hafaiedh, I. B., Gafsi, A., Yahyaoui, M. Y., & Aouinette, Y. (2024). A 
model-based approach for formal verification and performance 
evaluation of energy harvesting architectures in IoT systems: A case 
study of a long-term healthcare application. Simulation Modelling 
Practice and Theory, 136, 102990. 

Zhou, W., Zhao, Y., Zhang, Y., Wang, Y., & Yin, M. (2025). A 
comprehensive survey of UPPAAL‐assisted formal modeling and 
verification. Software: Practice and Experience, 55(2), 272-297. 

Nikitin, A., Iannucci, L., & Kaski, S. (2024). Tsgm: A flexible 
framework for generative modeling of synthetic time series. Advances 
in Neural Information Processing Systems, 37, 129042-129061. 



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

# **Synthetic Time Series Generation using ARIMA**

## 📋 Project Description

This project implements an ARIMA model to generate synthetic time series data,
The work was completed as part of a team project — this part was developed by **Fares ghezal**.

---

# 🔁 Repeatable Experiment: Step-by-Step Instructions

## Environment Setup

Install required libraries:

```bash
pip install kagglehub statsmodels numpy matplotlib scikit-learn plotly
```
Make sure you are using **Python 3.8+** 

---

# ⚙️ Technologies Used

- **Python 3.8+**
- **plotly**
- **statsmodels**
- **NumPy**
- **Matplotlib**
- **scikit-learn**

---

# 📈 Project Workflow

1. **Dataset import and analysis**  
import microsoft-stock-time-series-analysis from kagglehub and analyse dataset.
   
2. **Data Preprocessing**  
featue selection and reducing frequency`.

3. **Model Building**  
ARIMA model from statsmodels.

4. **Training**  
Model training on synthetic time series data.

6. **Evaluation**  
Plotting and analyzing the synthetic vs real data.

---

# 👤 Author

- **Fares Ghezal** — Part of a team project; contributed the ARIMA synthetic data generation .

---

# 📚 References

1. Hu C, Sun Z, Li C, Zhang Y, Xing C. Survey of time series data generation in IoT. Sensors. 2023
   
2. Chen P, Pedersen T, Bak-Jensen B, Chen Z. ARIMA-based time series model of stochastic wind power generation. IEEE transactions on power systems. 2009

3. Turowski M, Heidrich B, Weingärtner L, Springer L, Phipps K, Schäfer B, Mikut R, Hagenmeyer V. Generating synthetic energy time series: A review. Renewable and Sustainable Energy Reviews.
   
4. Scikit-learn Developers. (2024). "scikit-learn: Machine Learning in Python." [Link](https://scikit-learn.org/)


