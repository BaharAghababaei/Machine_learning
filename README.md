📚 **Machine Learning**

Welcome to the Machine Learning repository. This repository contains multiple projects demonstrating how machine learning techniques can be applied to real-world problems using Python, including experimentation in both local and cloud environments.


🔗 **1. Ride-Sharing Benefit Prediction**  
This project implements a deep learning model using PyTorch to predict the benefits of ride-sharing in terms of cost savings. The model is trained on a dataset containing the coordinates of two passengers and the shared-ride value, allowing it to estimate how much cost is saved when they share a trip. 

🧪 Key Features:

- Data preprocessing and normalization

- Neural network architecture design using PyTorch

- Hyperparameter tuning using Optuna

- Visualization of prediction errors

- Applied to the Kitsilano North-East area of Metro Vancouver (100 nodes, 308 edges)  

🐍 **Python Implementation (Jupyter Notebook)** → [View Notebook](https://github.com/baharaghababaei/Machine_learning/blob/main/docs/ride_sharing_prediction/Kitsilano_East.ipynb) 

---

🔗 **2. Heart Disease Prediction**  
This project builds a supervised machine learning workflow to classify whether a patient is likely to have heart disease based on various clinical features.
Multiple classifiers are tested, including Decision Tree, Random Forest, and XGBoost, with cross-validation and GridSearchCV used to optimize hyperparameters.

🧪 Key Features:

- Data cleaning, feature encoding (using one-hot encoding), and scaling

- Model evaluation using accuracy and cross-validation scores

- Hyperparameter tuning using GridSearchCV

- Visualizing the model performance using boxplots

**Dataset Features:**  
`Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`

🐍 **Python Implementation (Jupyter Notebook)** → [View Notebook](https://github.com/baharaghababaei/Machine_learning/blob/main/docs/Heart_disease_classification/Heart_Disease_Prediction.ipynb)    

---

🔗 **3. Diabetes Classification with Azure ML Pipelines**  

This project demonstrates how to build and automate a classification pipeline using Azure Machine Learning Studio. It classifies whether a patient is diabetic based on key clinical features. The full pipeline is developed with reusable components and MLflow tracking.

🧪 Key Features:
- Azure ML SDK v2 pipeline with modular components  
- Component 1: Preprocessing and scaling  
- Component 2: Model training with Logistic Regression, Random Forest, and XGBoost  
- Component 3: Model evaluation (ROC, AUC, Accuracy)  
- MLflow logging and ROC curve artifact tracking  
- Deployed and executed in Azure using remote compute targets  

**Dataset Features:**  
`Pregnancies`, `PlasmaGlucose`, `DiastolicBloodPressure`, `TricepsThickness`, `SerumInsulin`, `BMI`, `DiabetesPedigree`, `Age`

🐍 **Jupyter Script (pipeline + components)** → [`Diabetes_classification_pipeline.ipynb`](https://github.com/baharaghababaei/Machine_learning/blob/main/docs/diabetes_pipeline_Azure/Diabetes_classification_pipeline.ipynb)

---

### ✅ Environment

All projects are implemented in Python and use tools such as:
- Scikit-learn
- XGBoost
- PyTorch
- Optuna
- Azure ML SDK v2
- Pandas / NumPy / Matplotlib

---
