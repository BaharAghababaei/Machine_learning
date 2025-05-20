
# import libraries
import mlflow
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, dest="test_data")
    parser.add_argument("--model_path", type=str, dest="model_path")
    return parser.parse_args()

# function that reads the data
def load_data(test_data_path):

    test_df = pd.read_csv(Path(test_data_path) / "test.csv")

    X_test = test_df.drop("Diabetic", axis=1)
    y_test = test_df["Diabetic"]

    return X_test, y_test



def eval_model(model, X_test, y_test):

    model_name = type(model).__name__

    
    # calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.mean(y_hat == y_test)
    print(f"[{model_name}] Accuracy: {acc:.4f}")
    mlflow.log_metric("accuracy", acc)
    
    # calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print(f"[{model_name}] AUC: {auc:.4f}")
    mlflow.log_metric("auc", auc)
    
    # plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores[:,1])
    plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr,label=f"AUC = {auc:.2f}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend()

    output_file = f"ROC-Curve-{model_name}.png"
    plt.savefig(output_file)
    mlflow.log_artifact(output_file)

def main(args):
    # enable autologging
    mlflow.autolog(disable=True) 
    X_test, y_test = load_data(args.test_data)
    
    for model_dir in Path(args.model_path).iterdir():
        if model_dir.is_dir():
            print(f"Evaluating: {model_dir}")
            model = mlflow.sklearn.load_model(str(model_dir))
            eval_model(model, X_test, y_test)


if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)
    
    args = parse_args()
    main(args)
    
    print("*" * 60)
    print("\n\n")
