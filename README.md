# Adversarial Robustness: Defense against EOT Attacks

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Accuracy](https://img.shields.io/badge/Robustness-97%25-brightgreen)

> **A modernized implementation of adversarial defenses using Feature Distillation, migrated to TensorFlow 2.x.**

## 📄 Project Presentation
For a visual breakdown of the mathematical concepts, attack methodology, and defense architecture, please refer to the technical presentation:
👉 **[View Technical Presentation (PDF)](./Adversarial_robustness_defense.pdf)**

---

## 🔍 Overview
This project addresses the vulnerability of neural networks to **Obfuscated Gradients**, specifically focusing on breaking **Randomization Defenses** using **Expectation over Transformation (EOT)**.

Based on the seminal ICML 2018 paper *Obfuscated Gradients Give a False Sense of Security* (Athalye et al.), this project accomplishes two major engineering and research goals:
1.  **Modernization:** A complete reimplementation of the original paper's attack logic, migrated from legacy **TensorFlow 1.5** to modern **TensorFlow 2.x** (Eager Execution).
2.  **Novel Defense:** Development of a robust defense mechanism using **Feature Distillation** that withstands adaptive EOT attacks where standard randomization fails.

## 🚀 Results & Performance
Tested on **CIFAR-10** dataset. The baseline results confirm the findings of Athalye et al. that standard randomization offers **0% protection** against adaptive attacks. My proposed defense restores accuracy to near-clean levels.

| Defense Method | Attack Type | Accuracy (Post-Attack) | Notes |
| :--- | :--- | :--- | :--- |
| **Standard Randomization** | EOT (Adaptive) | **~0%** | Model collapses completely (Baseline) |
| **My Defense (Feature Distillation)** | EOT (Adaptive) | **97%** | **Maintains semantic consistency** |

> **Note:** Even in rare misclassification cases, the defense exhibits **semantic robustness**, with predictions remaining within the same object family (e.g., Cat $\to$ Coyote).

## 📂 Files in this Repository
* `Test_defense.py`: The main script to run the defense evaluation and attack simulations.
* `inceptionv3.py`: The TensorFlow 2.x implementation of the InceptionV3 model architecture used for the defense.
* `utils.py`: Utility functions for image processing, gradient estimation, and helper logic.
* `setup.sh`: Script for setting up the environment and downloading necessary checkpoints.
* `Adversarial_robustness_defense.pdf`: Technical slide deck explaining the theoretical proofs.
* `cat.jpg`: Sample image used for testing the defense pipeline.

## 🛠️ Methodology

### 1. The Attack: Expectation over Transformation (EOT)
Standard randomization defenses (resizing/padding) cause gradients to be stochastic. Attackers use EOT to estimate the true gradient by averaging over a distribution of transformations $T$:
$$\nabla \mathbb{E}_{t \sim T} [ f(t(x)) ]$$
This allows the attacker to "see through" the randomization.

### 2. The Defense: Feature Distillation
To counter this, I employed **Feature Distillation**. Instead of merely training on randomized labels, the defended model (Student) is trained to match the **internal feature representations** of a clean, pre-trained model (Teacher). This forces the network to learn high-level semantic features that are robust to low-level pixel perturbations introduced by the attacker.

## 💻 Installation & Usage

### Prerequisites
* Python 3.8+
* TensorFlow 2.x

### Setup
To replicate the results, you need to install the dependencies and download the required model checkpoints.

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/suparshwa31/adversarial-robustness-defense.git](https://github.com/suparshwa31/adversarial-robustness-defense.git)
    cd adversarial-robustness-defense
    ```

2.  **Install Python Dependencies**
    Install the required libraries (TensorFlow, NumPy, etc.) manually.
    ```bash
    pip install tensorflow numpy matplotlib pandas
    ```

3.  **Download Model Weights**
    Run the setup script to download the **InceptionV3** checkpoint and necessary data files.
    * *Note: This requires `wget` and `tar` (standard on Linux/Mac/Git Bash).*
    ```bash
    # Make the script executable
    chmod +x setup.sh
    
    # Run the script
    ./setup.sh
    ```

4.  **Run the Defense Evaluation**
    Once the data is downloaded into the `data/` directory, you can run the main evaluation script.
    ```bash
    python Test_defense.py
    ```

## 🔗 Reference & Attribution
This work is built upon the research and findings of Athalye et al. (ICML 2018).

* **Original Paper:** [Obfuscated Gradients Give a False Sense of Security (arXiv)](https://arxiv.org/pdf/1802.00420)
* **Original Codebase (TF 1.x):** [GitHub Link](https://github.com/anishathalye/obfuscated-gradients)
