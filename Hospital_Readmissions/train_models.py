# train_models.py
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline

def train_models(X_train, X_test, y_train, y_test, preprocessor):
   

    results = {}

    # === Logistic Regression ===
    log_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    log_params = {"clf__C": [0.01, 0.1, 1, 10]}
    log_grid = GridSearchCV(log_pipeline, log_params, cv=5, scoring="f1", n_jobs=-1)
    log_grid.fit(X_train, y_train)
    results["Logistic Regression"] = log_grid

    # === Random Forest ===
    rf_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(random_state=42))
    ])
    rf_params = {"clf__n_estimators": [100, 200],
                 "clf__max_depth": [5, 10, None]}
    rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring="f1", n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    results["Random Forest"] = rf_grid

    # === XGBoost ===
    xgb_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", xgb.XGBClassifier(
            eval_metric="logloss", 
            use_label_encoder=False, 
            random_state=42
        ))
    ])
    xgb_params = {"clf__n_estimators": [100, 200],
                  "clf__max_depth": [3, 5, 7]}
    xgb_grid = GridSearchCV(xgb_pipeline, xgb_params, cv=5, scoring="f1", n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    results["XGBoost"] = xgb_grid

    return results
