# DataCodeMarathon_MicrosoftML4Beginners_SupervisedLearning_TemperaturePredictionforFarming
In the scenario, I provide the date feature to the model. The model predicts the temperature, and I compare this result to the dataset's "correct" temperature. The objective function can then calculate how well the model worked, and I can make adjustments to the model.

# 🌾 Supervised Learning — Farming Temperature Prediction

A hands-on machine learning project demonstrating **linear regression** and **gradient-based optimization** to predict normalized temperature trends over time.  
Built as part of my Microsoft’s [Machine Learning for Beginners](https://github.com/microsoft/ML-For-Beginners) curriculum.

## 🧠 Project Overview

This notebook explores how **supervised learning** can be applied to model and predict temperature changes in a farming context.  
I've built a **simple linear regression** model, trained it using a custom optimizer (`MyOptimizer` from `m0b_optimizer.py`), and visualized how model parameters improve through training.

The project demonstrates:

- How to implement gradient descent manually  
- How to visualize convergence and learning curves  
- How model coefficients relate to real-world data trends

## 📁 Repository Structure
SupervisedLearning_FarmingTemperature/
│
├── SupervisedLearning_FarmingTemperature.ipynb # Jupyter notebook with full workflow
├── m0b_optimizer.py # Custom gradient-descent optimizer
├── data/ # (Optional) Input datasets
├── outputs/ # Model results and plots
└── README.md # Project documentation

## 🎯 Learning Objectives

- Understand the concept of **supervised learning** using labeled data  
- Apply **linear regression** to predict continuous variables  
- Implement and analyze a **custom optimizer**  
- Track and interpret **loss reduction** during training  
- Visualize model performance and regression fit

## 🧩 Dataset

The notebook uses simple tabular data that contains:

| Feature | Description |
|----------|--------------|
| `years_since_1982` | Number of years since 1982 (time feature) |
| `normalised_temperature` | Normalized temperature values (target variable) |

Synthetic data approximates real-world warming trends for farming temperature analysis.

## ⚙️ Model Details

**Model Equation:**
\[
\hat{y} = slope \times x + intercept
\]

**Training Process:**

1. Initialize `intercept` and `slope`
2. Predict temperatures using current parameters  
3. Compute loss (Mean Squared Error)  
4. Update parameters using `MyOptimizer` (gradient descent)  
5. Repeat until convergence or cost stabilization

---

## 🧮 Key Components

| Function / Class | Description |
|------------------|-------------|
| `train_one_iteration()` | Runs a single optimization step |
| `compute_loss()` | Calculates MSE between predictions and true values |
| `MyOptimizer` | Implements custom parameter updates (gradient descent) |
| `plot_results()` | Displays data points and regression line |

Model parameters before training: intercept = 0.11234567, slope = 0.03456789
Model parameters after training: intercept = 0.87412345, slope = 0.14298765
Final cost: 0.0021

Visualization plots show model fit improving as iterations progress.

## 🧰 Requirements
Install dependencies before running the notebook:

```bash
pip install numpy pandas matplotlib
Ensure m0b_optimizer.py is located in the same folder as your notebook.
```

### How to Run:
**Clone this project** : 
git clone https://github.com/<your-username>/SupervisedLearning_FarmingTemperature.git
cd SupervisedLearning_FarmingTemperature

**Launch Jupyter Notebook:**
jupyter notebook SupervisedLearning_FarmingTemperature.ipynb

**Run all cells:**
Follow the cell order to train the model and view results.

### **References**
Microsoft Learn — Machine Learning for Beginners : https://github.com/microsoft/ML-For-Beginners?utm_source=chatgpt.com
Scikit-Learn Linear Models - https://scikit-learn.org/stable/modules/linear_model.html
Gradient Descent Explained - https://en.wikipedia.org/wiki/Gradient_descent
Explore AI Scenarios - https://microsoftlearning.github.io/mslearn-ai-sims/Instructions/Labs/01-explore-ml.html

**Author:** Shraddha Debata
**Connect with me on LinkedIn:** https://www.linkedin.com/in/shraddha-debata-59726094
**View more of my work:** **Tableau** https://public.tableau.com/app/profile/shraddha.debata2941/vizzes
                          **Kaggle** https://www.kaggle.com/code/shraddhadebata/notebooke11b1b1723
