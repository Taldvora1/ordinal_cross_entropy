# Ordinal Cross Entropy (OCE)

This repository contains the implementation and experimental framework for **Ordinal Cross Entropy (OCE)**, a loss function designed for **ordinal classification problems**, where class labels have an inherent order (e.g., disease severity levels).

The project focuses on evaluating OCE in comparison to standard categorical and ordinal loss functions, particularly in medical image classification tasks.

## Project Overview

In many real-world classification problems, classes are **ordered** rather than nominal.
Standard categorical cross-entropy ignores this ordering and penalizes all misclassifications equally.

Ordinal Cross Entropy (OCE) incorporates:
- Ordinal structure between classes
- Distance-based misclassification penalties
- Flexible asymmetric and regularized variants

The framework allows systematic comparison between:
- Multy class Cross-Entropy
- Ordinal Loss (OL)
- Ordinal Cross Entropy (OCE)
- Regularized CE variants (Beta, Poisson, Binomial, Exponential)


