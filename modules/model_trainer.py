"""
Machine Learning Model Training, Comparison, Evaluation, and Explainable AI (XAI) Pipeline.
Trains regression models for DO, BOD, pH, EC, and classification models for WQI Category.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Feature sets for each regression target
FEATURE_SPECS = {
    "DO": {
        "features": ["Temperature", "pH", "Conductivity", "BOD", "Nitrate"],
        "unit": "mg/L",
        "description": "Dissolved Oxygen in water"
    },
    "BOD": {
        "features": ["Temperature", "pH", "Conductivity", "DO", "Nitrate"],
        "unit": "mg/L",
        "description": "Biochemical Oxygen Demand"
    },
    "pH": {
        "features": ["Temperature", "Conductivity", "DO", "BOD", "Nitrate"],
        "unit": "pH units",
        "description": "Potential of Hydrogen"
    },
    "Conductivity": {
        "features": ["Temperature", "pH", "DO", "BOD", "Nitrate"],
        "unit": "µmho/cm",
        "description": "Electrical Conductivity (EC)"
    }
}

CLASSIFICATION_FEATURES = ["Temperature", "DO", "pH", "Conductivity", "BOD", "Nitrate"]

def train_regression_models(df_clean, target_name="DO", test_size=0.2, random_state=42):
    """
    Trains and compares 4 regression algorithms for the given target.
    Returns: dictionary of models, metrics, predictions, and feature importances.
    """
    if target_name not in FEATURE_SPECS:
        raise ValueError(f"Target {target_name} not supported.")
        
    features = FEATURE_SPECS[target_name]["features"]
    
    # Clean dataset for target
    sub_df = df_clean.dropna(subset=[target_name] + features).copy()
    X = sub_df[features]
    y = sub_df[target_name]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    candidates = {
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=12, random_state=random_state),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=random_state),
        "Ridge Regression": Ridge(alpha=1.0)
    }
    
    results = {}
    best_model_name = None
    best_r2 = -999.0
    
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {
            "model": model,
            "metrics": {
                "R2": round(float(r2), 4),
                "MAE": round(float(mae), 4),
                "RMSE": round(float(rmse), 4)
            },
            "y_test": y_test.values,
            "y_pred": y_pred
        }
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            
    # Extract feature importance from best tree model (or Random Forest)
    imp_model = results.get("Random Forest")["model"]
    importances = pd.DataFrame({
        "Feature": features,
        "Importance": imp_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    # Save best model to disk
    best_info = results[best_model_name]
    save_path = os.path.join(MODELS_DIR, f"{target_name.lower()}_best_model.joblib")
    joblib.dump({
        "model": best_info["model"],
        "name": best_model_name,
        "features": features,
        "metrics": best_info["metrics"]
    }, save_path)
    
    return {
        "target": target_name,
        "results": results,
        "best_model_name": best_model_name,
        "best_metrics": best_info["metrics"],
        "feature_importances": importances,
        "X_test": X_test,
        "y_test": y_test
    }

def train_wqi_classifier(df_clean, test_size=0.2, random_state=42):
    """
    Trains and compares classification models for WQI Category.
    """
    sub_df = df_clean[df_clean["WQI_Category"].isin(["Good", "Poor", "Very Poor", "Unsuitable"])].copy()
    X = sub_df[CLASSIFICATION_FEATURES]
    y = sub_df["WQI_Category"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    candidates = {
        "Random Forest Classifier": RandomForestClassifier(n_estimators=120, max_depth=10, random_state=random_state),
        "XGBoost Classifier": xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }
    
    # Label mapping for XGBoost
    classes = sorted(y.unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for i, c in enumerate(classes)}
    
    results = {}
    best_model_name = None
    best_f1 = -1.0
    
    for name, model in candidates.items():
        if "XGBoost" in name:
            y_train_num = y_train.map(label_to_idx)
            y_test_num = y_test.map(label_to_idx)
            model.fit(X_train, y_train_num)
            preds_num = model.predict(X_test)
            preds = pd.Series(preds_num).map(idx_to_label).values
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=classes)
        
        results[name] = {
            "model": model,
            "metrics": {
                "Accuracy": round(float(acc), 4),
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "F1_Score": round(float(f1), 4)
            },
            "confusion_matrix": cm,
            "classes": classes,
            "y_test": y_test.values,
            "y_pred": preds
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            
    imp_model = results.get("Random Forest Classifier")["model"]
    importances = pd.DataFrame({
        "Feature": CLASSIFICATION_FEATURES,
        "Importance": imp_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    # Save model
    save_path = os.path.join(MODELS_DIR, "wqi_classifier_best.joblib")
    joblib.dump({
        "model": results[best_model_name]["model"],
        "name": best_model_name,
        "features": CLASSIFICATION_FEATURES,
        "classes": classes,
        "metrics": results[best_model_name]["metrics"]
    }, save_path)
    
    return {
        "results": results,
        "best_model_name": best_model_name,
        "best_metrics": results[best_model_name]["metrics"],
        "classes": classes,
        "feature_importances": importances,
        "X_test": X_test,
        "y_test": y_test
    }

def train_all_models(df_clean):
    """
    Convenience function to train all parameter predictors and classifier in one pass.
    """
    trained_artifacts = {}
    for target in ["DO", "BOD", "pH", "Conductivity"]:
        print(f"Training regression models for {target}...")
        trained_artifacts[target] = train_regression_models(df_clean, target_name=target)
        
    print("Training WQI Classification models...")
    trained_artifacts["WQI_Classifier"] = train_wqi_classifier(df_clean)
    print("All models successfully trained and cached.")
    return trained_artifacts
