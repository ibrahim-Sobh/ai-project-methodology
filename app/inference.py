import pandas as pd
import numpy as np
from app.preprocess import data_preprocessing
from joblib import load

MODEL_PATH = "../../models/clf.joblib"


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:

    # load the model
    model = load(MODEL_PATH)

    input_data = data_preprocessing(input_data, is_test=True)

    # Validation-set evaluation
    y_predictions = model.predict(input_data)

    return y_predictions
