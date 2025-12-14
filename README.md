🔭 Predicting Pulsar Stars – AI Project

This project was developed as part of the Artificial Intelligence course at the Faculty of Electrical Engineering, University of Sarajevo. The goal is to build an intelligent system that classifies whether a given celestial object is a pulsar or not, using machine learning techniques.



📌 Project Description

Pulsars are rapidly rotating neutron stars that emit beams of electromagnetic radiation. Accurately detecting pulsars is important in astrophysics, but the process is complicated by the presence of large amounts of noise and interference in astronomical data.

This project applies machine learning (ML) and artificial intelligence (AI) techniques to the HTRU2 dataset to develop a binary classification model capable of identifying pulsars from noise. The models are trained and evaluated using Python in Google Colab, with a focus on performance, interpretability, and handling class imbalance through resampling methods.





🧪 Dataset

Source: UCI Machine Learning Repository – HTRU2 Dataset

Instances: 17,898 total

Attributes: 8 numerical features per sample

Target: Binary classification (1 = pulsar, 0 = non-pulsar)

Class distribution: Highly imbalanced (≈9% pulsars)





Preprocessing:

Standardized all numeric features using StandardScaler

Used SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance

Dataset split: 80% training / 20% testing



⚙️ Technologies Used

Python 3

Scikit-learn

Pandas, NumPy

Matplotlib / Seaborn

TensorFlow / Keras (for neural network)

XGBoost (planned for extended testing)



🧠 Models and Algorithms

Support Vector Machine (SVM)

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree

Random Forest

Neural Network (Multi-layer Perceptron)

XGBoost



Evaluation Metrics:

Accuracy

Precision

Recall

F1-score

Special attention was given to Recall and F1-score due to the minority class being the true positive target (pulsars).



💡 Key Features

Handles class imbalance using synthetic oversampling (SMOTE)

Modular implementation for easy addition of new algorithms

Visualizations included: feature distributions, confusion matrices, ROC curves

Easy interpretability using scikit-learn and matplotlib

Link to presentation: https://www.canva.com/design/DAGqshQIGLA/rvklBj5QNBisTV3VQUy3Aw/edit



