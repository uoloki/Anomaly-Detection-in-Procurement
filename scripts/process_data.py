import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import os

def identify_potential_leaks(df, target_col='is_cartel', threshold_corr=0.8, threshold_auc=0.90):
    """
    Identifies potential data leakage features using correlation and single-feature predictive power.
    """
    leaks = []
    
    # 1. Correlation Check (Numeric)
    numeric_candidates = df.select_dtypes(include=['number']).columns.tolist()
    if target_col in numeric_candidates:
        corr_with_target = df[numeric_candidates].corrwith(df[target_col]).abs()
        suspicious_corr = corr_with_target[corr_with_target > threshold_corr]
        corr_leaks = suspicious_corr.drop(index=target_col, errors='ignore').index.tolist()
        leaks.extend(corr_leaks)
        if corr_leaks:
            print(f"Suspicious High Correlations: {corr_leaks}")

    # 2. Single Feature AUC Check
    suspicious_auc = []
    ids_to_ignore = ['persistent_id', 'tender_id', 'lot_id', 'buyer_id', 'bidder_id', 'cartel_id']
    
    for col in df.columns:
        if col == target_col or col in leaks or col in ids_to_ignore:
            continue
            
        try:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                temp_series = df[col].astype(str).fillna('missing')
                le = LabelEncoder()
                X_single = le.fit_transform(temp_series).reshape(-1, 1)
            else:
                X_single = df[col].fillna(0).values.reshape(-1, 1)
            
            clf = DecisionTreeClassifier(max_depth=2, random_state=42)
            clf.fit(X_single, df[target_col])
            probs = clf.predict_proba(X_single)[:, 1]
            auc_val = roc_auc_score(df[target_col], probs)
            
            if auc_val > threshold_auc:
                suspicious_auc.append(col)
                print(f"Suspicious feature detected: {col} (AUC: {auc_val:.4f})")
        except Exception:
            continue
            
    leaks.extend(suspicious_auc)
    
    # 3. Keyword Check
    keyword_leaks = [c for c in df.columns if ('cartel' in c.lower() or 'winner' in c.lower()) and c != target_col]
    leaks.extend(keyword_leaks)
    
    return list(set(leaks))

def process_data(input_path, output_path):
    print("--- Starting Data Processing ---")
    
    # Load data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file {input_path} not found.")
        return

    print(f"Original shape: {df.shape}")
    
    # Deduplicate
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Shape after deduplication: {df.shape}")
    
    # Define IDs and Known Metadata to drop
    known_ids = ['persistent_id', 'tender_id', 'lot_id', 'buyer_id', 'bidder_id', 'cartel_id']
    manual_exclusions = ['tender_year', 'cartel_tender']
    
    # Identify Leaks
    detected_leaks = identify_potential_leaks(df)
    
    # Combine all columns to drop
    cols_to_drop = list(set(known_ids + manual_exclusions + detected_leaks))
    
    # Keep essential columns for split/grouping if needed, but for "processed features" 
    # we usually want a clean dataset ready for training. 
    # However, model_selection.py needs 'tender_id' for GroupKFold.
    # So we will KEEP 'tender_id' in the processed file, but mark it to be dropped during training.
    
    # actually, model_selection.py loads the raw file and cleans it. 
    # To follow the plan "Refactor model_selection.py to Load data/processed_procurement_data.csv",
    # we should save a version that has leakage REMOVED but keeps 'tender_id' for grouping and 'is_cartel' for target.
    
    final_drop_list = [c for c in cols_to_drop if c != 'tender_id'] # Keep tender_id
    
    processed_df = df.drop(columns=[c for c in final_drop_list if c in df.columns], errors='ignore')
    
    print(f"Columns dropped: {final_drop_list}")
    print(f"Final processed shape: {processed_df.shape}")
    
    # Save
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    process_data(
        input_path="data/GTI_labelled_cartel_data_NOV2023.csv",
        output_path="data/processed_procurement_data.csv"
    )


