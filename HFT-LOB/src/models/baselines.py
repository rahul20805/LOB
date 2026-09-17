"""
Classical Machine Learning and Statistical Baselines (Table 2 & Section 4.3).
Includes Logistic Regression, Random Forest, XGBoost / GBDT with dropout, and SVM.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb


class BaselineModelZoo:
    """
    Factory and wrapper for classical ML baselines.
    """
    @staticmethod
    def create_model(model_name: str, **kwargs) -> Any:
        model_name = model_name.lower()
        if model_name in ["logistic", "logistic_regression", "lr"]:
            return LogisticRegression(
                C=kwargs.get("C", 1.0),
                class_weight="balanced",
                max_iter=kwargs.get("max_iter", 500),
                random_state=kwargs.get("random_state", 42)
            )
        elif model_name in ["rf", "random_forest"]:
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 8),
                class_weight="balanced",
                random_state=kwargs.get("random_state", 42),
                n_jobs=-1
            )
        elif model_name in ["xgb", "xgboost", "gbdt"]:
            return xgb.XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("lr", 0.05),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                random_state=kwargs.get("random_state", 42),
                eval_metric="mlogloss",
                n_jobs=-1
            )
        elif model_name in ["svm", "kernel_svm"]:
            return SVC(
                C=kwargs.get("C", 1.0),
                kernel=kwargs.get("kernel", "rbf"),
                probability=True,
                class_weight="balanced",
                random_state=kwargs.get("random_state", 42)
            )
        else:
            raise ValueError(f"Unknown baseline model: {model_name}")
