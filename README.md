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


