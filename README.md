📚 **Machine Learning**

This repository contains several projects demonstrating how machine learning techniques can be applied to real-world problems. These projects demonstrate different machine learning workflows, including deep learning, classical ML, time-series forecasting, and cloud-based ML pipelines.


🔗 **1. Ride-Sharing Benefit Prediction**  
This project implements a deep learning model using PyTorch to predict cost savings from ride-sharing. The model learns the relationship between passenger locations and the shared-ride benefit.

🧪 Key Features:

- Data preprocessing and normalization
- Neural network architecture design using PyTorch
- Hyperparameter tuning using Optuna
- Visualization of prediction errors
- Applied to the Kitsilano North-East area of Metro Vancouver (100 nodes, 308 edges)  

🐍 **Notebook** – [View Notebook](https://github.com/baharaghababaei/Machine_learning/blob/main/docs/ride_sharing_prediction/Kitsilano_East.ipynb)      

📄 **Ride-Sharing Data Analysis (PDF Report)** – [View Document](https://github.com/baharaghababaei/Machine_learning/blob/main/docs/ride_sharing_prediction/Ride-sharing%20Analysis.pdf)       

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

🐍 **Notebook** → [View Notebook](https://github.com/baharaghababaei/Machine_learning/blob/main/docs/Heart_disease_classification/Heart_Disease_Prediction.ipynb)    

---

🔗 **3. Diabetes Classification with Azure ML Pipelines**  

This project demonstrates how to build and automate a machine learning pipeline in Azure Machine Learning using reusable components. The pipeline classifies whether a patient is diabetic using clinical features.

🧪 Key Features:
- Azure ML SDK v2 pipeline with modular components  
- Component 1: Preprocessing and scaling  
- Component 2: Model training with Logistic Regression, Random Forest, and XGBoost  
- Component 3: Model evaluation (ROC, AUC, Accuracy)  
- MLflow logging and ROC curve artifact tracking  
- Executed in Azure using remote compute targets  

**Dataset Features:**  
`Pregnancies`, `PlasmaGlucose`, `DiastolicBloodPressure`, `TricepsThickness`, `SerumInsulin`, `BMI`, `DiabetesPedigree`, `Age`

🐍 **Notebook** → [`Diabetes_classification_pipeline.ipynb`](https://github.com/baharaghababaei/Machine_learning/blob/main/docs/diabetes_pipeline_Azure/Diabetes_classification_pipeline.ipynb)

---

🔗 **4. Traffic Travel Time Forecasting**

This project builds a multi-time-series forecasting model to predict travel time across 182 origin–destination (OD) pairs in Metro Vancouver using historical traffic data.

The goal is to generate day-ahead travel time forecasts (96 time steps) to support traffic planning and congestion analysis.

🧪 Key Features:

- Multi-series forecasting using skforecast
- Recursive forecasting with LightGBM
- Temporal feature engineering (hour, weekday, peak-hour indicators)
- Cyclical encoding of time features
- Rolling window statistics to capture short-term traffic dynamics
- Time-series backtesting for realistic model evaluation
- Comparison against a seasonal naive baseline
  
The model learns both temporal traffic patterns and network-level congestion behaviour to improve prediction accuracy.

🐍 **Notebook** → [View Notebook](https://github.com/baharaghababaei/Machine_learning/blob/main/docs/Time_Series_Forecasting/Travel_time_Forecasting.ipynb).

---
