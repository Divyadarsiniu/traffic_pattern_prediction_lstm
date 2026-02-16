# Real-Time Traffic Pattern Prediction using LSTM
## Overview

This project implements a Real-Time Traffic Pattern Prediction system using Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN) designed for time-series forecasting.
The model learns temporal dependencies in traffic data to predict future traffic volume and congestion levels, enabling intelligent traffic management and route optimization.

## Objective

Predict future traffic volume based on historical time-series data.

Capture temporal patterns such as peak hours and seasonal variations.

Build a deep learning model optimized for sequential data.

Enable real-time traffic forecasting for smart city applications.

## Why LSTM?

Traffic data is sequential and time-dependent. Traditional ML models fail to capture long-term dependencies.

### LSTM Advantages:

Handles long-term temporal dependencies

Avoids vanishing gradient problem

Suitable for time-series forecasting

Learns daily, weekly, and seasonal patterns
## Tech Stack

Language: Python 3.x

Libraries:

TensorFlow / Keras

NumPy

Pandas

Matplotlib

Scikit-learn

Deployment (Optional):

Flask / FastAPI

Streamlit

## Dataset Description

The dataset includes time-series traffic observations such as:
| Feature        | Description         |
| -------------- | ------------------- |
| timestamp      | Date and time       |
| traffic_volume | Number of vehicles  |      |
| holiday        | Holiday indicator   |
| day_of_week    | Day (0–6)           |

## Model Evaluation Metrics

RMSE (199.62), MAE (95.41), SMAPE (88.91%).

## Installation & Setup

step-1: git clone https://github.com/yourusername/real-time-traffic-lstm.git
cd real-time-traffic-lstm
step-2 python -m venv venv
source venv/bin/activate
step-3 pip install -r requirements.txt
step-4 python train.py
step-5 python predict.py

