# import libraries
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Random_State=42

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, dest="input_data")
    parser.add_argument("--output_data", type=str, dest="output_data")
    return parser.parse_args()

def get_data(input_path):

    df= pd.read_csv(input_path)
    return df


# remove missing values and duplicates
def clean_data(df):
    df = df.dropna()
    df=df.drop_duplicates()
    
    return df

# split data
def split_data(df):
    X= df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    'SerumInsulin','BMI','DiabetesPedigree','Age']].values
    y= df['Diabetic'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y ,random_state=Random_State)
    
    return X_train, X_test, y_train, y_test

# normalize data
def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled 


# main function
def main(args):
   
    df = get_data(args.input_data)

    cleaned_data = clean_data(df)

    X_train, X_test, y_train, y_test=split_data(cleaned_data)

    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
    
    # Save outputs as CSV files
    output_path = Path(args.output_data)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_names = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure',
                 'TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
    
    train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    train_df["Diabetic"] = y_train
    train_df.to_csv(output_path / "train.csv", index=False)

    test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    test_df["Diabetic"] = y_test
    test_df.to_csv(output_path / "test.csv", index=False)

if __name__ == "__main__":
    print("\n" + "*" * 60)
    args = parse_args()
    main(args)
    print("*" * 60 + "\n")
    
