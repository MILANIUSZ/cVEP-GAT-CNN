# cVEP-GAT-CNN
Graph-Attentive CNN for cVEP-BCI with Insights into Electrode Significance
(Accompanying repository for our IWANN 2025 submission)

Overview
This repository contains the code for the paper:

Graph-Attentive CNN for cVEP-BCI with Insights into Electrode Significance
by Milán András Fodor and Ivan Volosyak, submitted to IWANN 2025.

We propose a novel hybrid neural network that combines Convolutional Neural Networks (CNNs) and Graph Attention Networks (GATs) for code-modulated Visual Evoked Potential (cVEP) classification.
Our model achieves high validation accuracy (~94%) on a sample-wise classification task while providing insights into the most important EEG electrodes for cVEP decoding.

Key content:

A hybrid GAT-CNN model tailored for cVEP-based BCI systems

Insights into electrode significance using attention coefficients and permutation feature importance

Code for hyperparameter search using Bayesian optimization
