import pandas as pd

from src.dataset import FEATURE_COLUMNS
from src.preprocessing import prepare_train_test_split


def test_prepare_train_test_split_imputes_using_training_statistics_only():
    df = pd.DataFrame(
        [
            {"age": 45, "sex": 1, "cp": 4, "trestbps": 120, "chol": 180, "fbs": 0, "restecg": 0,
             "thalach": 170, "exang": 0, "oldpeak": 0.4, "slope": 1, "ca": 0, "thal": 3, "target": 0},
            {"age": 52, "sex": 0, "cp": 3, "trestbps": 130, "chol": 210, "fbs": 0, "restecg": 1,
             "thalach": 160, "exang": 1, "oldpeak": 1.2, "slope": 2, "ca": 1, "thal": 7, "target": 1},
            {"age": 60, "sex": 1, "cp": 4, "trestbps": 140, "chol": 230, "fbs": 1, "restecg": 2,
             "thalach": 140, "exang": 1, "oldpeak": 2.0, "slope": 2, "ca": 2, "thal": 7, "target": 1},
            {"age": 48, "sex": 0, "cp": 2, "trestbps": 122, "chol": None, "fbs": 0, "restecg": 0,
             "thalach": 165, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3, "target": 0},
            {"age": 63, "sex": 1, "cp": 4, "trestbps": 150, "chol": None, "fbs": 0, "restecg": 2,
             "thalach": 120, "exang": 1, "oldpeak": 3.1, "slope": 2, "ca": 2, "thal": 7, "target": 1},
            {"age": 36, "sex": 0, "cp": 1, "trestbps": 110, "chol": 190, "fbs": 0, "restecg": 0,
             "thalach": 175, "exang": 0, "oldpeak": 0.1, "slope": 1, "ca": 0, "thal": 3, "target": 0},
        ]
    )

    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df,
        test_size=0.33,
        random_state=42,
        impute_missing=True,
    )

    assert list(X_train.columns) == FEATURE_COLUMNS
    assert list(X_test.columns) == FEATURE_COLUMNS
    assert not X_train.isnull().any().any()
    assert not X_test.isnull().any().any()
    assert len(X_train) + len(X_test) == len(df)
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})
