import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_features(df):
    df = df.copy()

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["HasFamily"] = (df["FamilySize"] > 1).astype(int)

    df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")[0].str.strip()

    df["Title"] = df["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    })

    common_titles = ["Mr", "Miss", "Mrs", "Master"]
    df.loc[~df["Title"].isin(common_titles), "Title"] = "Rare"

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 18, 35, 55, np.inf],
        labels=["kid", "young_adult", "adult", "senior"],
        right=False,
    )

    df["AgeGroup"] = df["AgeGroup"].astype("object").fillna("Unknown")

    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")

    df["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
    df["Sex_AgeGroup"] = df["Sex"].astype(str) + "_" + df["AgeGroup"].astype(str)

    return df


def load_train_data(config):
    df = pd.read_csv(config.paths.train_data)
    df = add_features(df)

    y = df[config.data.target].copy()

    X = df.drop(columns=[
        config.data.target,
        config.data.id_column,
        "Name",
        "Ticket",
        "Cabin",
    ])

    return X, y


def load_test_data(config):
    df = pd.read_csv(config.paths.test_data)
    df = add_features(df)

    test_ids = df[config.data.id_column].copy()

    X_test = df.drop(columns=[
        config.data.id_column,
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