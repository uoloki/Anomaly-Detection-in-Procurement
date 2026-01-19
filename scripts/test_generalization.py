"""
Generalization Test: Can the model predict cartels for UNSEEN BIDDERS?

The core question: Is the model learning cartel BEHAVIOR or just memorizing
which bidders are cartels?

Test 1: Split by BIDDER (train on some bidders, test on new bidders)
Test 2: Split by COUNTRY (train on 6 countries, test on 1 held-out country)
Test 3: Compare with tender-split (original approach)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

def build_pipeline(X):
    """Build preprocessing + model pipeline"""
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    transformers = []
    if numeric_cols:
        transformers.append(('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_cols))
    
    if categorical_cols:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )
    
    return Pipeline([('preprocessor', preprocessor), ('classifier', model)])


def test_generalization(raw_path, processed_path):
    print("="*70)
    print("GENERALIZATION TEST: Does the model truly learn cartel behavior?")
    print("="*70)
    
    # Load data
    raw_df = pd.read_csv(raw_path).drop_duplicates().reset_index(drop=True)
    processed_df = pd.read_csv(processed_path)
    
    # Get bidder_id from raw (aligned after dedup)
    processed_df['bidder_id'] = raw_df['bidder_id'].values
    processed_df['country'] = raw_df['country'].values
    
    target_col = 'is_cartel'
    y = processed_df[target_col]
    
    # Features (exclude identifiers)
    feature_cols = [c for c in processed_df.columns 
                   if c not in [target_col, 'tender_id', 'bidder_id', 'country']]
    X = processed_df[feature_cols]
    
    results = []
    
    # =========================================================================
    # TEST 1: Original Tender-Split (our current approach)
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 1: Split by TENDER (current approach)")
    print("-"*70)
    
    groups_tender = processed_df['tender_id']
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    aucs = []
    for train_idx, test_idx in sgkf.split(X, y, groups_tender):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipeline = build_pipeline(X_train)
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, y_prob))
    
    tender_auc = np.mean(aucs)
    tender_std = np.std(aucs)
    print(f"ROC AUC: {tender_auc:.4f} ± {tender_std:.4f}")
    results.append({'Split': 'By Tender', 'ROC AUC': tender_auc, 'Std': tender_std})
    
    # =========================================================================
    # TEST 2: Split by BIDDER (the true test!)
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 2: Split by BIDDER (test on UNSEEN bidders)")
    print("-"*70)
    print("This tests if we're learning behavior vs memorizing bidder identity")
    
    groups_bidder = processed_df['bidder_id']
    
    # Can't use stratified with bidder groups easily, use GroupKFold
    gkf = GroupKFold(n_splits=5)
    
    aucs = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups_bidder)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Check class balance in fold
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        
        pipeline = build_pipeline(X_train)
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        try:
            auc = roc_auc_score(y_test, y_prob)
            aucs.append(auc)
            print(f"  Fold {fold+1}: AUC={auc:.4f} | Train cartel={train_rate:.1%}, Test cartel={test_rate:.1%}")
        except ValueError as e:
            print(f"  Fold {fold+1}: Skipped (only one class in test set)")
    
    if aucs:
        bidder_auc = np.mean(aucs)
        bidder_std = np.std(aucs)
        print(f"\nROC AUC: {bidder_auc:.4f} ± {bidder_std:.4f}")
        results.append({'Split': 'By Bidder', 'ROC AUC': bidder_auc, 'Std': bidder_std})
    
    # =========================================================================
    # TEST 3: Leave-One-Country-Out
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 3: Leave-One-Country-Out (train on 6, test on 1)")
    print("-"*70)
    print("Tests geographic generalization")
    
    countries = processed_df['country'].unique()
    country_aucs = []
    
    for held_out in countries:
        train_mask = processed_df['country'] != held_out
        test_mask = processed_df['country'] == held_out
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        if len(y_test.unique()) < 2:
            print(f"  {held_out}: Skipped (only one class)")
            continue
        
        pipeline = build_pipeline(X_train)
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        n_test = len(y_test)
        cartel_rate = y_test.mean()
        
        print(f"  {held_out}: AUC={auc:.4f} | n={n_test}, cartel={cartel_rate:.1%}")
        country_aucs.append(auc)
    
    if country_aucs:
        country_auc = np.mean(country_aucs)
        country_std = np.std(country_aucs)
        print(f"\nMean ROC AUC: {country_auc:.4f} ± {country_std:.4f}")
        results.append({'Split': 'Leave-Country-Out', 'ROC AUC': country_auc, 'Std': country_std})
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("GENERALIZATION TEST RESULTS")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    print("""
INTERPRETATION:

If 'By Bidder' AUC << 'By Tender' AUC:
   → Model is memorizing bidder identities, not learning behavior
   → Performance will NOT generalize to new bidders
   → The 99% AUC is misleading

If 'Leave-Country-Out' AUC is low:
   → Model is learning country-specific patterns
   → May not generalize to new jurisdictions

WHAT REALISTIC PERFORMANCE LOOKS LIKE:
   - By Bidder: 0.70-0.85 AUC (if learning true patterns)
   - Leave-Country-Out: 0.65-0.80 AUC (cross-border generalization is hard)
    """)
    
    # Calculate drop
    if len(results) >= 2:
        tender_auc = results[0]['ROC AUC']
        bidder_auc = results[1]['ROC AUC']
        drop = tender_auc - bidder_auc
        drop_pct = (drop / tender_auc) * 100
        
        print(f"\nPERFORMANCE DROP (Tender → Bidder split): {drop:.4f} ({drop_pct:.1f}%)")
        
        if drop_pct > 15:
            print("⚠️  SIGNIFICANT DROP: Model is likely memorizing bidder identities!")
        elif drop_pct > 5:
            print("⚠️  MODERATE DROP: Some bidder memorization occurring")
        else:
            print("✓ MINIMAL DROP: Model appears to learn generalizable patterns")
    
    return results_df

if __name__ == "__main__":
    test_generalization(
        "data/GTI_labelled_cartel_data_NOV2023.csv",
        "data/processed_procurement_data.csv"
    )


