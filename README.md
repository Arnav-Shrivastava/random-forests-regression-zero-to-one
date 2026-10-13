# Random Forests: Zero to Confident

A comprehensive, beginner-friendly learning project demystifying Random Forests from first mathematical principles to robust implementation, empirical validation, hyperparameter tuning, interpretability, and failure mode analysis.

## Project Overview
- **Objective**: Master Random Forests conceptually, mathematically, and practically.
- **Dataset**: Heart Disease Risk Prediction (UCI Heart Disease / Scikit-Learn fallback).
- **Core Methodology**: Moving from single decision trees $\to$ Bagging $\to$ Random Forests with feature subsampling, verifying theoretical predictions at every stage.

## Repository Contents
- `random_forests_zero_to_one.ipynb`: Complete runnable learning notebook with step-by-step theory, math derivations, code implementations, interactive experiments, and practice challenges.
- `requirements.txt`: Python environment dependencies.
- `README.md`: Project summary, learning outcomes, key results, and reproduction instructions.

## Roadmap & Milestones
- [ ] Step 1: Theory of Ensembles, Variance Reduction, and Decorrelation
- [ ] Step 2: From Scratch Implementation (Bagging vs Random Forest)
- [ ] Step 3: Empirical Correlation Experiment Proving the Theory
- [ ] Step 4: Hyperparameter Tuning with Scikit-Learn RandomizedSearchCV
- [ ] Step 5: Out-of-Bag (OOB) Error Mechanics and Validation
- [ ] Step 6: Feature Importance Analysis (MDI vs Permutation vs SHAP)
- [ ] Step 7: Tree Count Scaling: Asymptotic Convergence Without Overfitting
- [ ] Step 8: Critical Failure Modes and Boundary Limitations
- [ ] Step 9: Mastery Check: "Prove It" Questions, Exercises, and Extensions

## Getting Started
```bash
pip install -r requirements.txt
jupyter lab random_forests_zero_to_one.ipynb
```
