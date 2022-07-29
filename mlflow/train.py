
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from preprocess import data_preprocessing
from joblib import dump
import mlflow
import mlflow.sklearn
import sys

DATA_PATH = "../data/train/train.csv"

def data_split_test_train(data: pd.DataFrame,
                                    test_size: int = 0.2)-> pd.DataFrame:
    # Split Train / Test
    X = data.loc[:, data.columns != 'Response']
    y = data.Response
    # First Split L between Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=42)
    # return all splitted data sets ( 4 sets )
    return X_train, X_test, y_train, y_test
    
    
    return evaluations_dict

if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test =data_split_test_train(data)
    X_train = data_preprocessing(X_train, is_test=False)

    # Define an evaluation dictonary
    evaluations_dict = dict()
    #input
    max_depth = int(sys.argv[1])
    n_estimators = int(sys.argv[2]) 

    with mlflow.start_run():
        clf= RandomForestClassifier( max_depth= max_depth,n_estimators= n_estimators)

        # Train model
        clf.fit(X_train, y_train)

        # Model Build Evalution on Testing Set
        # -------------------------------------
        # Preprocessing(cleaning data and using trained encoders,scalars)
        X_test = data_preprocessing(X_test, is_test=True)

        # Testing-set evaluation
        y_pred = clf.predict(X_test)
    
        y_pred = y_pred.ravel()
        y_pred = abs(y_pred)
        y_true = y_test.ravel()

        key = "Accuracy"
        accuracy = accuracy_score(y_true, y_pred)
        
        key = "F1"
        f1 = f1_score(y_true, y_pred, average='weighted')
    
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1_Score", f1)
        mlflow.sklearn.log_model(clf, "model")