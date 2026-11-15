# -----------------------------------------
# Creation of Model Builders
# -----------------------------------------

# For RandomForest
def build_rf(cfg, y_train):
    if cfg["model"].get("class_weight") == "balanced":
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
    else:
        class_weights = None

    return RandomForestClassifier(
        n_estimators=cfg["model"]["n_estimators"],
        max_depth=cfg["model"]["max_depth"],
        class_weight=class_weights,
        random_state=cfg["model"]["random_state"]
    )

# -----------------------------------------
# For XGBoost
# -----------------------------------------
from xgboost import XGBClassifier

# -----------------------------------------
# Importation of necessary dependencies
# -----------------------------------------
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

def build_xgb(cfg, y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)

    return XGBClassifier(
        n_estimators=cfg["model"]["n_estimators"],
        max_depth=cfg["model"]["max_depth"],
        learning_rate=cfg["model"]["learning_rate"],
        subsample=cfg["model"].get("subsample", 1.0),
        colsample_bytree=cfg["model"].get("colsample_bytree", 1.0),
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=cfg["model"]["random_state"]
    )
# -----------------------------------------
# For RUSBoost
# -----------------------------------------
from imblearn.ensemble import RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def build_rusboost(cfg, y_train):
    base_estimator = DecisionTreeClassifier(
        max_depth=cfg["model"].get("base_max_depth", 1),
        random_state=cfg["model"]["random_state"]
    )

    return RUSBoostClassifier(
        estimator=base_estimator,
        n_estimators=cfg["model"]["n_estimators"],
        learning_rate=cfg["model"]["learning_rate"],
        random_state=cfg["model"]["random_state"]
    )