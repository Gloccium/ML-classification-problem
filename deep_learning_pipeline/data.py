import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TitanicDataset(Dataset):
    def __init__(self, X, y=None):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.X = torch.from_numpy(X)

        if y is not None:
            y = np.ascontiguousarray(np.asarray(y), dtype=np.float32)
            self.y = torch.from_numpy(y)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]

        return self.X[idx]


def add_features(df):
    df = df.copy()

    # New feature for family size and whether a person has a family
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["HasFamily"] = (df["FamilySize"] > 1).astype(int)

    # Extracting title from name and labeling rare titles under one value "rare"
    df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")[0].str.strip()

    df["Title"] = df["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    })

    common_titles = ["Mr", "Miss", "Mrs", "Master"]
    df.loc[~df["Title"].isin(common_titles), "Title"] = "Rare"

    # Creating age groups (0-18, 18-35, 35-55, 55+)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 18, 35, 55, np.inf],
        labels=["kid", "young_adult", "adult", "senior"],
        right=False,
    )

    df["AgeGroup"] = df["AgeGroup"].astype("object").fillna("Unknown")

    # Cabin features
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")

    # Complex features
    df["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
    df["Sex_AgeGroup"] = df["Sex"].astype(str) + "_" + df["AgeGroup"].astype(str)

    return df


def load_train_data(config):
    train_df = pd.read_csv(config.paths.train_data)
    train_df = add_features(train_df)

    target = config.data.target
    id_column = config.data.id_column

    y = train_df[target].copy()

    X = train_df.drop(columns=[
        target,
        id_column,
        "Name",
        "Ticket",
        "Cabin",
    ])

    return X, y


def load_test_data(config):
    test_df = pd.read_csv(config.paths.test_data)
    test_df = add_features(test_df)

    id_column = config.data.id_column

    test_ids = test_df[id_column].copy()

    X_test = test_df.drop(columns=[
        id_column,
        "Name",
        "Ticket",
        "Cabin",
    ])

    return X_test, test_ids


def create_preprocessor(X_train):
    numerical_columns = X_train.select_dtypes(include=["number"]).columns
    categorical_columns = X_train.select_dtypes(exclude=["number"]).columns

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])


def create_fold_dataloaders(X, y, train_idx, val_idx, config):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    preprocessor = create_preprocessor(X_train)

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)

    train_dataset = TitanicDataset(X_train, y_train)
    val_dataset = TitanicDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    input_size = X_train.shape[1]

    return train_loader, val_loader, preprocessor, input_size


def create_inference_loader(X, preprocessor, batch_size):
    X = preprocessor.transform(X)
    dataset = TitanicDataset(X)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
