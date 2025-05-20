# import libraries
import mlflow
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



Random_State=42
def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_data", dest='training_data', type=str) 
    parser.add_argument("--model_output", dest='model_output', type=str) 

    args = parser.parse_args()

    return args


# function that reads the data
def load_data(training_data_path):

    train_df = pd.read_csv(Path(training_data_path) / "train.csv")
    
    X_train = train_df.drop("Diabetic", axis=1)
    y_train = train_df["Diabetic"]


    return X_train, y_train



def train_LR(X_train, y_train,output_path ):
    with mlflow.start_run(nested=True):
        mlflow.log_param("model_type", "Logistic Regression")
        print("Training model...")
        model= LogisticRegression(C=(1/0.01), solver="liblinear").fit(X_train, y_train)
        mlflow.sklearn.save_model(model, output_path / "logistic_regression")

    return model

def train_RF(X_train, y_train,output_path ):
    with mlflow.start_run(nested=True):
        mlflow.log_param("model_type", "Random Forest")
        print("Training model...")
        model=RandomForestClassifier(max_depth= 8, min_samples_split= 2, n_estimators= 500 , random_state = Random_State).fit(X_train, y_train)
        mlflow.sklearn.save_model(model, output_path / "random_forest")
        
    return model


def train_XG(X_train, y_train,output_path ):
    with mlflow.start_run(nested=True):
        mlflow.log_param("model_type", "XGBoost")
        print("Training model...")
        model=XGBClassifier(learning_rate= 0.01, max_depth= 2, n_estimators=500 , random_state = Random_State).fit(X_train, y_train)
        mlflow.sklearn.save_model(model, output_path / "xgboost")
        
    return model




def main(args):
    
    # read data
    X_train, y_train = load_data(args.training_data)
    # path
    output_path = Path(args.model_output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # train model
    with mlflow.start_run(): #parent run
        train_LR(X_train, y_train, output_path)
        train_RF(X_train, y_train, output_path)
        train_XG(X_train, y_train, output_path)
       

    

    
if __name__ == "__main__":
    print("\n" + "*" * 60)
    args = parse_args()
    main(args)
    print("*" * 60 + "\n")
