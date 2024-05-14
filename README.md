# Interpreting Art-Induced Emotions via Concept Bottleneck Models

## Overview
This repository houses the MSc AI thesis project titled "Interpreting Art-Induced Emotions via Concept Bottleneck Models." The primary goal of this project is to investigate what triggers specific emotions in humans when viewing artworks. Utilizing the ArtEmis dataset and concept bottleneck models, this research explores the intricate relationship between art and emotional responses.

## Directory Structure
Below is an overview of the repository's structure, providing insights into the contents and functionalities of each directory:

- **`2000_subset_data/`**: Contains a subset of 2000 data points from the ArtEmis dataset, featuring multiple variations for detailed analysis.
- **`cmd/`**: Includes executable scripts for concept extraction, network training, and other command-line utilities.
- **`concept_processing/`**: Contains the logic for the old CoDEx pipeline, adapted for concept processing within the scope of this project.
- **`data_analysis/`**: Scripts and utilities for preprocessing and analyzing the ArtEmis data.
- **`jupiter-notebooks/thesis-experiments/`**: Jupyter notebooks used for conducting experiments such as producing attention boards and computing Mutual Information (MI) graphs.
- **`logs/`**: Stores neural network-based learning training logs, which document the progression and results of model training.
- **`nn/`**: Source code related to neural network training, including model architectures and ablation studies.

## Running the Project

To run specific components of the project, navigate to the corresponding directory and execute the scripts provided. For example:

- **Run Concept Extraction**

  ```bash
  python cmd/run_concept_extraction.py

  ```bash
  python nn/train_model.py









