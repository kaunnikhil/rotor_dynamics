#  Real-Time Digital Twin of a 4-Disk Hollow Flexible Shaft Rotor

> *What if we could predict a rotor's future before it starts vibrating?*

A high-fidelity Digital Twin framework that combines **Rotor Dynamics**, **Finite Element Analysis (FEA)**, and **Machine Learning** to predict the dynamic behavior of a multi-disk flexible rotor system in real time.

Instead of solving computationally expensive physics equations every time operating conditions change, this project trains an AI surrogate model capable of instantly estimating:

- Critical Speeds
- Natural Frequencies
- Mode Shapes
- Resonance Regions
- Vibration Amplitudes

The result is a Digital Twin that thinks like a physics simulator but responds at the speed of machine learning.

---

## Project Motivation

Rotating machinery powers everything from aircraft engines and turbines to industrial compressors and manufacturing equipment.

The challenge?

Traditional rotor dynamic simulations are computationally intensive and unsuitable for real-time monitoring.

This project bridges that gap by creating a Digital Twin that:

* Learns from thousands of FEM simulations
* Predicts rotor behavior in milliseconds
* Enables predictive maintenance strategies
* Demonstrates how AI can accelerate classical mechanical engineering workflows

Think of it as teaching a machine-learning model to become a rotor dynamics expert.

---

#  System Architecture

```text
Rotor Geometry
      │
      ▼
Finite Element Model
      │
      ▼
10,000+ Simulations
      │
      ▼
Dataset Generation
      │
      ▼
Gaussian Process Regression
      │
      ▼
Real-Time Predictions
      │
      ▼
Interactive Digital Twin Dashboard
```

---

# Core Features

### High-Fidelity Rotor Dynamics Model

The physics engine models a:

* Four-disk rotor system
* Hollow flexible shaft
* Multiple operating speeds
* Dynamic vibration behavior

Using Finite Element Methods, the framework computes:

* Mass Matrix (M)
* Damping Matrix (C)
* Gyroscopic Matrix (G)
* Stiffness Matrix (K)

to solve the governing rotor dynamic equations.

---

###  Campbell Diagram 

Automatically generates:

* Natural frequency plots
* Speed-frequency intersections
* Critical speed identification
* Resonance visualization

A fundamental tool used in rotor design and vibration analysis.

---

###  Machine Learning Surrogate Model

Instead of solving large eigenvalue systems repeatedly, the project trains a:

**Gaussian Process Regression (GPR)** model

Advantages:

* Fast predictions
* Excellent performance on limited engineering datasets
* Built-in uncertainty estimation
* Physics-informed decision support

---

###  Real-Time Digital Twin

The trained surrogate model acts as a virtual replica of the rotor.

Given new operating parameters, the Digital Twin instantly predicts:

* Dynamic response
* Critical speed shifts
* Resonance risks
* Expected vibration amplitudes

without rerunning expensive simulations.

---

###  Interactive Dashboard

Built using Streamlit and Plotly.

Features:

* Parameter sliders
* Live predictions
* Interactive Campbell diagrams
* Vibration trend visualization
* Uncertainty-aware ML outputs

---

# Tech Stack

## Mechanical Engineering

* Finite Element Method (FEM)
* Rotor Dynamics
* Modal Analysis
* Vibration Analysis
* Campbell Diagram Analysis

## Data Science & Machine Learning

* NumPy
* Pandas
* SciPy
* Scikit-Learn
* Gaussian Process Regression

## Digital Twin Layer

* Streamlit
* Plotly

## Development

* Git
* GitHub
* Jupyter Notebook

---

# Engineering Concepts Demonstrated

This project showcases practical applications of:

* Rotor Dynamics
* Structural Vibrations
* Finite Element Analysis
* Digital Twin Technology
* Surrogate Modeling
* Gaussian Process Regression
* Predictive Maintenance
* Mechanical AI Systems

---

## Author

**Nikhil Menaria**
B.Tech Mechanical Engineering
Indian Institute of Technology (ISM) Dhanbad

