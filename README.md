# Dueling Deep Q-Network Implementation Report

## Introduction

This report presents the implementation of a **Dueling Deep Q-Network (Dueling DQN)** using PyTorch. The Dueling DQN architecture enhances the traditional DQN by separately estimating the state-value and advantage functions, leading to more stable and efficient learning in reinforcement learning tasks.

## Installation and Running

Follow these steps to set up the project on your local machine:

1. Platform

   - Ubuntu 20.04 LTS

2. Create and activate the conda environment

```py
conda env create -f environment.yml -n group_37
conda activate group_37
```

3. Run the code of our **Neural Networks**

```py
python3 ./main_Network.py
```

4. Remove The Env

```py
conda remove -n group_37 --all
```

## Tools and Libraries

The project leverages the following Python libraries:

- **PyTorch:** For building and training the neural network models.
- **Matplotlib:** For plotting training metrics and visualizations.
- **NumPy:** For numerical operations and data manipulation.
- **Graphviz:** For visualizing the computational graph of the neural network.
