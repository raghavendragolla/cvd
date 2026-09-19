"""
Next-Gen Machine Learning Model Zoo & Clinical Classifier Architectures.
Supports Stratified 5-Fold CV, GridSearchCV Hyperparameter Optimization,
Cost-Sensitive Class Weighting, Calibrated Ensembles, and Multi-Scale Architectures.
"""

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


def get_model_grid_search_configs(random_state: int = 42, use_class_weights: bool = False) -> dict:
    """
    Returns base estimators and focused hyperparameter grids for 9 base models,
    designed for high-efficiency Stratified 5-Fold GridSearchCV.
    """
    cw = "balanced" if use_class_weights else None
    scale_pos = 1.0 if not use_class_weights else 1.15

    configs = {
        "⚡ XGBoost (Primary Tree)": {
            "estimator": XGBClassifier(
                random_state=random_state,
                eval_metric="logloss",
                scale_pos_weight=scale_pos
            ),
            "param_grid": {
                "n_estimators": [80, 120],
                "max_depth": [3, 4],
                "learning_rate": [0.03, 0.08],
                "subsample": [0.8, 1.0]
            }
        },
        "🐱 CatBoost (Clinical Booster)": {
            "estimator": CatBoostClassifier(
                random_seed=random_state,
                auto_class_weights="Balanced" if use_class_weights else None,
                verbose=0
            ),
            "param_grid": {
                "iterations": [100, 150],
                "depth": [3, 5],
                "learning_rate": [0.03, 0.08]
            }
        },
        "💡 LightGBM (Fast Booster)": {
            "estimator": LGBMClassifier(
                random_state=random_state,
                class_weight=cw,
                verbose=-1
            ),
            "param_grid": {
                "n_estimators": [80, 120],
                "max_depth": [3, 5],
                "learning_rate": [0.03, 0.08],
                "num_leaves": [15, 31]
            }
        },
        "🌲 Random Forest Ensemble": {
            "estimator": RandomForestClassifier(
                random_state=random_state,
                class_weight=cw
            ),
            "param_grid": {
                "n_estimators": [100, 150],
                "max_depth": [4, 6],
                "min_samples_split": [2, 4],
                "min_samples_leaf": [1, 2]
            }
        },
        "🌳 Extra Trees Classifier": {
            "estimator": ExtraTreesClassifier(
                random_state=random_state,
                class_weight=cw
            ),
            "param_grid": {
                "n_estimators": [100, 150],
                "max_depth": [4, 6],
                "min_samples_split": [2, 4]
            }
        },
        "📈 Gradient Boosting (GBDT)": {
            "estimator": GradientBoostingClassifier(
                random_state=random_state
            ),
            "param_grid": {
                "n_estimators": [80, 120],
                "max_depth": [3, 4],
                "learning_rate": [0.03, 0.08]
            }
        },
        "🎯 Support Vector Machine (SVC)": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, class_weight=cw, random_state=random_state))
            ]),
            "param_grid": {
                "clf__C": [0.5, 1.0, 2.0],
                "clf__kernel": ["rbf", "linear"]
            }
        },
        "📐 Logistic Regression (Linear)": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight=cw, solver="liblinear", random_state=random_state))
            ]),
            "param_grid": {
                "clf__C": [0.2, 0.8, 1.5],
                "clf__penalty": ["l1", "l2"]
            }
        },
        "🧠 Multi-Layer Perceptron (Neural Net)": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(max_iter=400, random_state=random_state))
            ]),
            "param_grid": {
                "clf__hidden_layer_sizes": [(32, 16), (64, 32)],
                "clf__alpha": [0.005, 0.05]
            }
        }
    }
    return configs


def build_super_ensemble(tuned_estimators: dict) -> VotingClassifier:
    """
    Builds the 10th model: a Super-Ensemble Soft-Voting Classifier combining
    the top tuned estimators.
    """
    selected_estimators = []
    keys_map = {
        "xgb": "⚡ XGBoost (Primary Tree)",
        "cat": "🐱 CatBoost (Clinical Booster)",
        "lgbm": "💡 LightGBM (Fast Booster)",
        "rf": "🌲 Random Forest Ensemble",
        "lr": "📐 Logistic Regression (Linear)"
    }
    
    for short_key, full_key in keys_map.items():
        if full_key in tuned_estimators:
            selected_estimators.append((short_key, tuned_estimators[full_key]))
            
    if not selected_estimators:
        # Fallback to taking the first 4 tuned models
        for i, (k, est) in enumerate(list(tuned_estimators.items())[:4]):
            selected_estimators.append((f"m{i}", est))
            
    voting_ensemble = VotingClassifier(
        estimators=selected_estimators,
        voting="soft"
    )
    return voting_ensemble


def get_model_zoo(random_state: int = 42, use_class_weights: bool = False) -> dict:
    """
    Returns default tuned models (backward compatibility for direct usage).
    """
    cw = "balanced" if use_class_weights else None
    scale_pos = 1.0 if not use_class_weights else 1.15
    
    xgb = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos,
        random_state=random_state,
        eval_metric="logloss"
    )

    lgbm = LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.8,
        class_weight=cw,
        random_state=random_state,
        verbose=-1
    )

    catboost = CatBoostClassifier(
        iterations=150,
        depth=4,
        learning_rate=0.05,
        auto_class_weights="Balanced" if use_class_weights else None,
        random_seed=random_state,
        verbose=0
    )

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight=cw,
        random_state=random_state
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_split=4,
        class_weight=cw,
        random_state=random_state
    )

    gbdt = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=random_state
    )

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.8,
            penalty="l2",
            solver="liblinear",
            class_weight=cw,
            random_state=random_state
        ))
    ])

    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            class_weight=cw,
            random_state=random_state
        ))
    ])

    mlp_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=400,
            alpha=0.01,
            random_state=random_state
        ))
    ])

    voting_ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb),
            ("cat", catboost),
            ("lgbm", lgbm),
            ("rf", rf),
            ("lr", lr_pipe)
        ],
        voting="soft"
    )

    models = {
        "🫀 Super-Ensemble (Top Precision)": voting_ensemble,
        "⚡ XGBoost (Primary Tree)": xgb,
        "🐱 CatBoost (Clinical Booster)": catboost,
        "💡 LightGBM (Fast Booster)": lgbm,
        "🌲 Random Forest Ensemble": rf,
        "🌳 Extra Trees Classifier": extra_trees,
        "📈 Gradient Boosting (GBDT)": gbdt,
        "🧠 Multi-Layer Perceptron (Neural Net)": mlp_pipe,
        "🎯 Support Vector Machine (SVC)": svm_pipe,
        "📐 Logistic Regression (Linear)": lr_pipe
    }
    return models


def build_calibrated_model(base_estimator, method: str = "sigmoid", cv: int = 5):
    """
    Wraps a base model into CalibratedClassifierCV using either:
    - 'sigmoid' (Platt Scaling)
    - 'isotonic' (Isotonic Regression)
    """
    return CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)
