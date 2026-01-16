from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class Hyperparameter:
    name: str
    type: str  # 'float', 'int', 'choice'
    default: Any
    options: List[Any] = field(default_factory=list)  # For 'choice'
    min: float | None = None
    max: float | None = None


@dataclass
class ModelDefinition:
    name: str
    cls: Type[BaseEstimator]
    hyperparameters: List[Hyperparameter]


MODELS = {
    "Logistic Regression": ModelDefinition(
        name="Logistic Regression",
        cls=LogisticRegression,
        hyperparameters=[
            Hyperparameter(name="C", type="float", default=1.0, min=0.01, max=100.0),
            Hyperparameter(
                name="solver",
                type="choice",
                default="liblinear",
                options=["liblinear", "lbfgs", "newton-cg", "sag", "saga"],
            ),
        ],
    ),
    "Random Forest": ModelDefinition(
        name="Random Forest",
        cls=RandomForestClassifier,
        hyperparameters=[
            Hyperparameter(name="n_estimators", type="int", default=100, min=10, max=500),
            Hyperparameter(name="max_depth", type="int", default=10, min=1, max=100),
            Hyperparameter(
                name="criterion",
                type="choice",
                default="gini",
                options=["gini", "entropy", "log_loss"],
            ),
        ],
    ),
}
