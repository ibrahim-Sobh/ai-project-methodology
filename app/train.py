
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from app.preprocess import data_preprocessing
from joblib import dump

MODEL_PATH = "../../models/clf.joblib"

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


def build_model(data: pd.DataFrame) -> dict[str, str]:

    # split data into Train, Test, and Validation
    X_train, X_test, y_train, y_test =data_split_test_train(data)

    # Preprocessing(cleaning data and training encoders,scalars)
    X_train = data_preprocessing(X_train, is_test=False)

    # Define an evaluation dictonary
    evaluations_dict = dict()

    # Defining the Machine Learning model
    clf= RandomForestClassifier( max_depth= 10,n_estimators= 30, random_state= 42)

    # Train model
    clf.fit(X_train, y_train)
    dump(clf, MODEL_PATH)

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
    evaluations_dict.update( dict({key: accuracy}))
    
    key = "F1"
    f1 = f1_score(y_true, y_pred, average='weighted')
    evaluations_dict.update( dict({key: f1}))
    
    return evaluations_dict
