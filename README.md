# Informational Fairness Study - Technical Implementation

This repository contains the technical implementation of a study on informational fairness. The study focuses on training and evaluating fair and unfair classification models using a home loan application dataset. The final models are utilized to predict outcomes for three exemplary instances, with each instance accompanied by different sets of explanations.

## Explanations in the Study

The study presents two sets of explanations:
- **Baseline Explanations**: Local feature importance & demographics.
- **Advanced Explanations**: Counterfactuals & normative explanations.

The explanations are generated for selected instances using:
- **DiCE**: For generating counterfactual explanations.
- **LIME**: For generating local feature importance.

## Getting Started

### Prerequisites

To run the adversarial models, set up the environment described in the `environment.yml` file using Conda:

```bash 
conda env create -f environment.yml
```

For all other files, ensure you have the required packages installed by setting up the environment described in requirements.txt:

```bash
pip install -r requirements.txt
```

## Repository Structure

### Notebooks

- **Adversarial_classifier_undersampled_females.ipynb**:
  - Development of the fair model.
  - Generation of counterfactuals and local feature importance for selected fair predictions.

- **MLP_undersampling_females.ipynb**:
  - Development of the unfair model.
  - Generation of counterfactuals and local feature importance for selected unfair predictions.

- **Fairness Audit Notebooks**:
  - Notebooks with the prefix `Fairness_audit_` evaluate various models. Each notebook focuses on a specific model, analyzing its fairness.

### Benchmarking

- **Benchmark_counterfactual.ipynb**:
  - Compares sets of counterfactuals generated for the purposes of contestability or recourse, focusing on the predictions of fair and unfair models.

### Master Thesis

The detailed findings and discussion of the study can be found in the master thesis:
- **MA_informational_fairness.pdf**


