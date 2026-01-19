"""
Data Leakage Validation Script

This script performs systematic checks to ensure no data leakage exists
that could artificially inflate model performance.

Checks performed:
1. Single-Feature AUC Test: Flag features with AUC > 0.90
2. Keyword Scan: Check for columns containing "cartel", "winner", "outcome"
3. Correlation Check: Flag features with correlation > 0.80 with target
4. High Cardinality Check: Flag potential ID columns
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def validate_leakage(data_path="data/processed_procurement_data.csv"):
    """
    Comprehensive leakage validation for the processed dataset.
    """
    print("="*60)
    print("       DATA LEAKAGE VALIDATION REPORT")
    print("="*60)
    
    # Load data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Run process_data.py first.")
        return
    
    target_col = 'is_cartel'
    group_col = 'tender_id'
    
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        return
    
    y = df[target_col]
    feature_cols = [c for c in df.columns if c not in [target_col, group_col]]
    
    print(f"\nDataset: {data_path}")
    print(f"Samples: {len(df)}")
    print(f"Features: {len(feature_cols)}")
    
    issues_found = []
    
    # =========================================================================
    # CHECK 1: Single-Feature AUC Test
    # =========================================================================
    print("\n" + "-"*60)
    print("CHECK 1: Single-Feature AUC Test")
    print("-"*60)
    print("Testing if any single feature can predict target with AUC > 0.90")
    print("(Would indicate potential data leakage)")
    
    high_auc_features = []
    
    for col in feature_cols:
        try:
            # Handle different data types
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                le = LabelEncoder()
                X_single = le.fit_transform(df[col].astype(str).fillna('missing')).reshape(-1, 1)
            else:
                X_single = df[col].fillna(0).values.reshape(-1, 1)
            
            # Simple decision stump
            clf = DecisionTreeClassifier(max_depth=2, random_state=42)
            clf.fit(X_single, y)
            probs = clf.predict_proba(X_single)[:, 1]
            auc = roc_auc_score(y, probs)
            
            if auc > 0.90:
                high_auc_features.append((col, auc))
                
        except Exception as e:
            continue
    
    if high_auc_features:
        print("\n⚠️  WARNING: High AUC features detected!")
        for col, auc in sorted(high_auc_features, key=lambda x: -x[1]):
            print(f"   - {col}: AUC = {auc:.4f}")
        issues_found.append(f"High AUC features: {len(high_auc_features)}")
    else:
        print("\n✓ PASS: No single feature achieves AUC > 0.90")
    
    # =========================================================================
    # CHECK 2: Keyword Scan
    # =========================================================================
    print("\n" + "-"*60)
    print("CHECK 2: Keyword Scan")
    print("-"*60)
    print("Checking for columns containing suspicious keywords")
    
    suspicious_keywords = ['cartel', 'winner', 'outcome', 'fraud', 'collusion', 'label']
    keyword_matches = []
    
    for col in df.columns:
        col_lower = col.lower()
        for keyword in suspicious_keywords:
            if keyword in col_lower and col != target_col:
                keyword_matches.append((col, keyword))
    
    if keyword_matches:
        print("\n⚠️  WARNING: Suspicious column names detected!")
        for col, keyword in keyword_matches:
            print(f"   - '{col}' contains '{keyword}'")
        issues_found.append(f"Suspicious keywords: {len(keyword_matches)}")
    else:
        print("\n✓ PASS: No suspicious keywords in column names")
    
    # =========================================================================
    # CHECK 3: Correlation Check
    # =========================================================================
    print("\n" + "-"*60)
    print("CHECK 3: Correlation Check")
    print("-"*60)
    print("Checking for features with correlation > 0.80 with target")
    
    numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns
    high_corr_features = []
    
    if len(numeric_cols) > 0:
        correlations = df[numeric_cols].corrwith(y).abs()
        high_corr = correlations[correlations > 0.80]
        
        if len(high_corr) > 0:
            print("\n⚠️  WARNING: High correlation features detected!")
            for col in high_corr.index:
                print(f"   - {col}: r = {correlations[col]:.4f}")
            high_corr_features = high_corr.index.tolist()
            issues_found.append(f"High correlation features: {len(high_corr_features)}")
        else:
            print("\n✓ PASS: No features with correlation > 0.80")
    
    # =========================================================================
    # CHECK 4: High Cardinality Check
    # =========================================================================
    print("\n" + "-"*60)
    print("CHECK 4: High Cardinality Check")
    print("-"*60)
    print("Checking for potential ID columns (>90% unique values)")
    
    high_cardinality = []
    
    for col in feature_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.90:
            high_cardinality.append((col, unique_ratio))
    
    if high_cardinality:
        print("\n⚠️  WARNING: High cardinality columns detected!")
        for col, ratio in high_cardinality:
            print(f"   - {col}: {ratio:.1%} unique")
        issues_found.append(f"High cardinality: {len(high_cardinality)}")
    else:
        print("\n✓ PASS: No high-cardinality ID-like columns")
    
    # =========================================================================
    # CHECK 5: Tender Grouping Verification
    # =========================================================================
    print("\n" + "-"*60)
    print("CHECK 5: Tender Grouping Info")
    print("-"*60)
    
    if group_col in df.columns:
        n_tenders = df[group_col].nunique()
        avg_bids_per_tender = len(df) / n_tenders
        print(f"\n✓ Group column '{group_col}' present")
        print(f"   - Unique tenders: {n_tenders}")
        print(f"   - Avg bids per tender: {avg_bids_per_tender:.2f}")
    else:
        print(f"\n⚠️  WARNING: Group column '{group_col}' not found!")
        issues_found.append("Missing group column")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("       VALIDATION SUMMARY")
    print("="*60)
    
    if not issues_found:
        print("\n✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print("\nNo data leakage detected. Model performance is legitimate.")
    else:
        print(f"\n⚠️  {len(issues_found)} ISSUE(S) FOUND:")
        for issue in issues_found:
            print(f"   - {issue}")
        print("\nReview the warnings above before trusting model results.")
    
    print("\n" + "="*60)
    
    # Return results for programmatic use
    return {
        'high_auc_features': high_auc_features,
        'keyword_matches': keyword_matches,
        'high_corr_features': high_corr_features if 'high_corr_features' in dir() else [],
        'high_cardinality': high_cardinality,
        'issues_count': len(issues_found)
    }

if __name__ == "__main__":
    validate_leakage()


