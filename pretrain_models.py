import os
import sys
# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_processing import load_and_preprocess_data, get_clean_modeling_dataset
from modules.model_trainer import train_all_models

print("Starting model training pipeline...")
df = load_and_preprocess_data()
print(f"Loaded dataset: {df.shape}")
df_clean = get_clean_modeling_dataset(df)
print(f"Cleaned modeling subset: {df_clean.shape}")

artifacts = train_all_models(df_clean)
print("Model training completed successfully!")
for k, v in artifacts.items():
    if "best_model_name" in v:
        print(f"Target: {k} | Best Model: {v['best_model_name']} | Metrics: {v['best_metrics']}")
