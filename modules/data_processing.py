"""
Data Preprocessing, Validation, and Feature Engineering Pipeline for CPCB River Water Quality Data.
Handles missing values, outliers, units, conversions, and metadata extraction.
"""

import os
import pandas as pd
import numpy as np
from modules.wqi_engine import compute_row_wqi, get_wqi_classification, classify_cpcb_pwqc

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cpcb_water_quality_cleaned.csv")

def load_and_preprocess_data(file_path=DATA_PATH):
    """
    Loads raw extracted CPCB dataset, computes WQI, CPCB class, derived parameters,
    and returns a fully preprocessed DataFrame.
    """
    if not os.path.exists(file_path):
        # Fallback to local
        file_path = "cpcb_water_quality_cleaned.csv"
        
    df = pd.read_csv(file_path)
    
    # 1. Clean and normalize string fields
    df["State"] = df["State"].fillna("Not Specified").str.strip()
    df["River"] = df["River"].fillna("Unknown River").str.strip()
    df["Location"] = df["Location"].fillna("Unknown Location").str.strip()
    df["District"] = df["District"].fillna("Not Specified").str.strip()
    df["Station_Code"] = df["Station_Code"].astype(str)
    
    # 2. Derive TDS from Electrical Conductivity (Conductivity in µmho/cm)
    # Limnological standard conversion: TDS (mg/L) ≈ 0.64 * EC (µS/cm)
    df["TDS"] = df["Conductivity"].apply(lambda ec: round(ec * 0.64, 2) if pd.notnull(ec) else np.nan)
    
    # 3. Chemical Oxygen Demand (COD) Limnological Ratio
    # Typical unpolluted to moderately treated Indian river COD:BOD ratio is ~2.0 - 2.5
    df["COD_Estimated"] = df["BOD"].apply(lambda bod: round(bod * 2.1, 2) if pd.notnull(bod) else np.nan)
    
    # 4. Compute Weighted Arithmetic WQI and Sub-indices
    wqi_list = []
    wqi_cat_list = []
    cpcb_class_list = []
    
    for idx, row in df.iterrows():
        wqi_val, _ = compute_row_wqi(row)
        wqi_list.append(wqi_val)
        cat_info = get_wqi_classification(wqi_val)
        wqi_cat_list.append(cat_info["category"])
        
        cpcb_info = classify_cpcb_pwqc(
            do=row.get("DO"),
            ph=row.get("pH"),
            bod=row.get("BOD"),
            fc=row.get("Fecal_Coliform")
        )
        cpcb_class_list.append(cpcb_info["class_code"])
        
    df["WQI"] = wqi_list
    df["WQI_Category"] = wqi_cat_list
    df["CPCB_Class"] = cpcb_class_list
    
    return df

def get_missing_value_summary(df):
    """
    Generates structured missing values report.
    """
    summary = []
    for col in df.columns:
        n_missing = df[col].isnull().sum()
        pct_missing = (n_missing / len(df)) * 100.0
        summary.append({
            "Column": col,
            "Missing_Count": int(n_missing),
            "Missing_Percentage": round(pct_missing, 2),
            "Present_Count": int(len(df) - n_missing),
            "Data_Type": str(df[col].dtype)
        })
    return pd.DataFrame(summary).sort_values(by="Missing_Count", ascending=False)

def detect_outliers_iqr(df, column):
    """
    Detects outliers using standard 1.5 * IQR criterion.
    """
    series = df[column].dropna()
    if len(series) == 0:
        return 0, np.nan, np.nan, pd.Series()
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    lower_bound = max(0.0, q25 - 1.5 * iqr)
    upper_bound = q75 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers), round(lower_bound, 2), round(upper_bound, 2), outliers

def get_clean_modeling_dataset(df):
    """
    Returns robust modeling dataframe with median imputation for non-target inputs
    and filtering of extreme recording errors.
    """
    df_clean = df.dropna(subset=["DO", "pH", "BOD"]).copy()
    
    # Impute missing features with medians
    for col in ["Temperature", "Conductivity", "Nitrate"]:
        med = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(med)
        
    # Standard environmental bounds
    df_clean = df_clean[
        (df_clean["pH"] >= 4.0) & (df_clean["pH"] <= 11.0) &
        (df_clean["DO"] >= 0.0) & (df_clean["DO"] <= 20.0) &
        (df_clean["BOD"] >= 0.0) & (df_clean["BOD"] <= 150.0) &
        (df_clean["Conductivity"] >= 10.0) & (df_clean["Conductivity"] <= 20000.0)
    ]
    return df_clean
