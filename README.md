# Real-Time Digital Twin of a 4-Disk Hollow Flexible Shaft Rotor

> *What if we could predict a rotor's future before it starts vibrating?*

### Live Digital Twin Demo

**Try it here:**  
http://rotordynamics-hw2sf7cppvfuubffellqgk.streamlit.app/

*Adjust rotor geometry, disk mass, and bearing stiffness to see real-time predictions of natural frequencies and operational behavior without running a full finite element simulation.*

A high-fidelity Digital Twin framework that combines **Rotor Dynamics**, **Finite Element Analysis (FEA)**, and **Machine Learning** to predict the dynamic behavior of a multi-disk flexible rotor system in real time.

Instead of solving computationally expensive physics equations every time operating conditions change, this project trains a Machine Learning surrogate model capable of instantly estimating:

- Critical Speeds
- Natural Frequencies
- Mode Shapes
- Resonance Regions
- Vibration Amplitudes

The result is a Digital Twin that captures the behavior of a physics-based simulation while delivering predictions in milliseconds.

---

## Project Motivation

Rotating machinery powers everything from aircraft engines and turbines to industrial compressors and manufacturing equipment.

The challenge?

Traditional rotor dynamic simulations are computationally intensive and unsuitable for real-time monitoring.

This project bridges that gap by creating a Digital Twin that:

* Learns from thousands of FEM simulations
* Predicts rotor behavior in milliseconds
* Enables predictive maintenance strategies
* Demonstrates how Machine Learning can accelerate classical mechanical engineering workflows

Think of it as teaching a machine-learning model the behavior of a rotor dynamics simulator.

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

