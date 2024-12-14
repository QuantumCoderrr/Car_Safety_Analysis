import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(data_path):
    """
    Load the dataset from the specified path and preprocess it.
    """
    data = pd.read_csv(data_path, header=0)
    return data

def encode_data(data):
    """
    Encode categorical variables using LabelEncoder.
    """
    label_encoders = {}
    for column in data.columns[:-1]:  # Exclude the 'class' column from encoding
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Encode the target column 'class' separately
    le_target = LabelEncoder()
    data["class"] = le_target.fit_transform(data["class"])
    label_encoders["class"] = le_target

    return data, label_encoders
