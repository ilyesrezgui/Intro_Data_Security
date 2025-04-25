# 🧪 Synthetic Data Generation with UPPAAL – SEIHR Model

This repository presents a formal approach to synthetic data generation using **UPPAAL** and **model checking**, based on the SEIHR epidemiological model:

- **S**: Susceptible  
- **E**: Exposed  
- **I**: Infected  
- **H**: Hospitalized  
- **R**: Recovered

Using **UPPAAL timed automata**, we simulate the logical transitions of individuals through these states to generate high-quality synthetic data and verify system behavior.

---

## 📌 Motivation

Epidemiological data is often limited or sensitive. With UPPAAL, we can:
- Simulate realistic disease spread scenarios.
- Generate traceable and explainable synthetic datasets.
- Ensure logical consistency of transitions using formal verification.

---

## 📊 SEIHR Model Logic

| State         | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| Susceptible   | Population starts here. Decreases over time as individuals get exposed.     |
| Exposed       | Transition state before becoming infectious.                                |
| Infected      | Active disease stage. Can lead to hospitalization or recovery.              |
| Hospitalized  | A subset of infected individuals require care. Can recover or deteriorate.  |
| Recovered     | Final stage for recovered individuals. Increases over time.                 |

> Example: The number of hospitalized individuals can fluctuate, while susceptible individuals generally decrease as the disease spreads or a cure is discovered.

---

## ✅ Model Checking

We verify critical logical and temporal properties such as:

- **Safety**:  
  ```text
  A[] (H > 0 imply I > 0)
