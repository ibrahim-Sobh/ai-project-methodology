import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from joblib import dump, load

ENCODER_PATH = "../models/encoder.joblib"

def drop_columns(data: pd.DataFrame,
                 to_remove_columns: list = []) -> pd.DataFrame:
    return data.drop(to_remove_columns, axis=1)


def encode_categorical_features(encoder, data: pd.DataFrame,
                                is_test: bool = False) -> pd.DataFrame:
    data_categorical = data.select_dtypes(include=['object']).columns
    if not is_test:
        encoder.fit(data[data_categorical])
        dump(encoder, ENCODER_PATH)
    data[data_categorical] = encoder.transform(data[data_categorical])
    return data


def fill_features_nulls(data: pd.DataFrame) -> pd.DataFrame:

    data_numerical = data.select_dtypes([np.int64, np.float64]).columns
    data_categorical = data.select_dtypes(include=['object']).columns

    data[data_numerical] = data[data_numerical].fillna(
        data[data_numerical].mean())

    for feature in data_categorical:
        data[feature].interpolate(
            method='linear', limit_direction='forward', inplace=True)
        data[feature].interpolate(
            method='linear', limit_direction='backward', inplace=True)

    return data

def data_preprocessing(data: pd.DataFrame,
                       is_test: bool = False) -> pd.DataFrame:

    if is_test:
        # load the encoder
        encoder = load(ENCODER_PATH)
    else:
        # Create Encoder
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan)

    # Carefully Selected Features (after analysis)
    list_of_features =['Family_Hist_3', 'Product_Info_2',
                       'Family_Hist_4', 'Medical_History_2',
                       'Employment_Info_6', 'Ht', 'Medical_History_1',
                       'Employment_Info_1', 'Id', 'Ins_Age',
                       'Product_Info_4', 'Wt', 'BMI']


    data = data[list_of_features]
    
    data = fill_features_nulls(data)

    data = encode_categorical_features(encoder, data, is_test)

    return data
