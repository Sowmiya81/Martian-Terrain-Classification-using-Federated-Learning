# Martian-Terrain-Classification-using-Federated-Learning

# Overview
This project focuses on the multi-class classification of Martian terrain using federated learning and the DenseNet-121 architecture. The goal is to classify Martian landmass into seven distinct classes: crater, dark dune, slope streak, bright dune, impact ejecta, swiss cheese, and spider. Accurate classification provides valuable insights into Martian geological processes and aids in mission planning, including selecting landing sites and plotting safe exploration routes.

# Key Features
1. Multi-class classification of Martian terrain into 7 distinct categories.
2. Utilizes federated learning, preserving data privacy while enabling decentralized model training across multiple data sources.
3. Implements the DenseNet-121 architecture for robust feature extraction and high classification accuracy.
4. The dataset used for training is the HiRISE dataset, obtained from the Mars Reconnaissance Orbiter.

# Dataset
HiRISE Dataset: High-resolution images of the Martian surface, collected by the Mars Reconnaissance Orbiter.
The dataset is divided across different sources, simulating the decentralized nature of federated learning.

# Model Architecture
DenseNet-121: A Convolutional Neural Network (CNN) architecture that leverages dense connections to enhance feature propagation and reduce vanishing gradient issues.
The federated learning framework enables model training across distributed datasets without requiring data centralization, ensuring data privacy.

# Training
Federated Learning: Model is trained collaboratively across multiple data sources, using a client-server architecture.
Each client trains the model locally on its dataset, and only model updates are shared with the server.
Extensive hyperparameter tuning was performed to optimize the model’s performance.

# Results
The federated DenseNet-121 model demonstrates high accuracy and robust performance in classifying Martian terrain.
This approach lays the foundation for efficient terrain classification in future space exploration missions, contributing to our understanding of the Martian surface.
