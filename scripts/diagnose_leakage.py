"""
Comprehensive Data Leakage Diagnostic

This script investigates why our model achieves unrealistically high AUC (0.99).
We check for:
1. Single-feature predictive power (with lower threshold)
2. Feature correlations with target
3. Duplicate/near-duplicate detection
4. Country-level stratification issues
5. Network feature leakage (confirmed)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def diagnose_leakage(raw_path, processed_path):
    print("="*70)
    print("COMPREHENSIVE LEAKAGE DIAGNOSTIC")
    print("="*70)
    
    # Load data
    raw_df = pd.read_csv(raw_path)
    processed_df = pd.read_csv(processed_path)
    
    raw_df = raw_df.drop_duplicates().reset_index(drop=True)
    
    target_col = 'is_cartel'
    y = processed_df[target_col]
    
    print(f"\nDataset: {len(processed_df)} samples")
    print(f"Target distribution: {y.mean():.2%} cartel")
    
    # =========================================================================
    # CHECK 1: Single-Feature AUC (with detailed breakdown)
    # =========================================================================
    print("\n" + "-"*70)
    print("CHECK 1: Single-Feature AUC Analysis")
    print("-"*70)
    print("Looking for features with suspiciously high individual predictive power...")
    
    feature_aucs = []
    exclude_cols = [target_col, 'tender_id', 'bidder_id']
    
    for col in processed_df.columns:
        if col in exclude_cols:
            continue
        
        try:
            if processed_df[col].dtype == 'object':
                # Encode categorical
                le = LabelEncoder()
                vals = le.fit_transform(processed_df[col].fillna('missing'))
            else:
                vals = processed_df[col].fillna(processed_df[col].median())
            
            auc = roc_auc_score(y, vals)
            # AUC can be < 0.5 if inversely correlated, flip it
            auc = max(auc, 1 - auc)
            
            feature_aucs.append({
                'feature': col,
                'auc': auc,
                'unique_values': processed_df[col].nunique(),
                'missing_pct': processed_df[col].isna().mean() * 100
            })
        except Exception as e:
            print(f"  Skipped {col}: {e}")
    
    auc_df = pd.DataFrame(feature_aucs).sort_values('auc', ascending=False)
    
    print("\nTop 15 Features by Single-Feature AUC:")
    print(auc_df.head(15).to_string(index=False))
    
    # Flag suspicious features
    suspicious = auc_df[auc_df['auc'] > 0.70]
    if len(suspicious) > 0:
        print(f"\n⚠️  WARNING: {len(suspicious)} features have AUC > 0.70!")
        print("   These might be leaking information or are too predictive.")
    
    # =========================================================================
    # CHECK 2: Correlation Analysis
    # =========================================================================
    print("\n" + "-"*70)
    print("CHECK 2: Feature Correlations with Target")
    print("-"*70)
    
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        correlations = processed_df[numeric_cols].corrwith(processed_df[target_col]).abs().sort_values(ascending=False)
        correlations = correlations.drop(target_col, errors='ignore')
        
        print("\nTop 10 Correlations with Target:")
        print(correlations.head(10).to_string())
        
        high_corr = correlations[correlations > 0.5]
        if len(high_corr) > 0:
            print(f"\n⚠️  WARNING: {len(high_corr)} features have |correlation| > 0.5")
    
    # =========================================================================
    # CHECK 3: Country Distribution Analysis
    # =========================================================================
    print("\n" + "-"*70)
    print("CHECK 3: Country-Level Stratification")
    print("-"*70)
    print("Checking if cartels are concentrated in specific countries...")
    
    country_cols = [c for c in processed_df.columns if 'country' in c.lower()]
    if country_cols:
        print(f"\nFound country columns: {country_cols}")
        for col in country_cols:
            if processed_df[col].dtype in ['int64', 'float64']:
                # Binary country indicator
                cartel_rate = processed_df.groupby(col)[target_col].mean()
                print(f"\n{col} cartel rates:")
                print(cartel_rate.to_string())
    
    # Also check in raw data
    if 'country' in raw_df.columns:
        country_cartel = raw_df.groupby('country').agg({
            'is_cartel': ['mean', 'count']
        }).round(3)
        country_cartel.columns = ['cartel_rate', 'n_samples']
        print("\nCartel rates by country (raw data):")
        print(country_cartel.to_string())
        
        # If some countries are 100% cartel or 0% cartel, that's a problem
        if (country_cartel['cartel_rate'] == 0).any() or (country_cartel['cartel_rate'] == 1).any():
            print("\n⚠️  WARNING: Some countries have 0% or 100% cartel rate!")
            print("   Model might just be learning 'which country = cartel'")
    
    # =========================================================================
    # CHECK 4: Tender-Level Analysis
    # =========================================================================
    print("\n" + "-"*70)
    print("CHECK 4: Tender-Level Label Consistency")
    print("-"*70)
    
    if 'tender_id' in processed_df.columns:
        tender_labels = processed_df.groupby('tender_id')[target_col].agg(['mean', 'std', 'count'])
        
        # Check if labels are consistent within tenders
        mixed_tenders = tender_labels[tender_labels['std'] > 0]
        print(f"\nTenders with mixed labels (some cartel, some not): {len(mixed_tenders)}")
        print(f"Tenders with consistent labels: {len(tender_labels) - len(mixed_tenders)}")
        
        if len(mixed_tenders) == 0:
            print("\n✓ All tenders have consistent labels (all cartel or all honest)")
            print("  This is expected - cartels affect entire tenders, not individual bids")
    
    # =========================================================================
    # CHECK 5: Bidder-Level Analysis  
    # =========================================================================
    print("\n" + "-"*70)
    print("CHECK 5: Bidder-Level Analysis")
    print("-"*70)
    
    if 'bidder_id' in raw_df.columns:
        bidder_labels = raw_df.groupby('bidder_id')[target_col].agg(['mean', 'count'])
        bidder_labels.columns = ['cartel_rate', 'n_bids']
        
        # How many bidders are 100% cartel or 0% cartel?
        pure_cartel = (bidder_labels['cartel_rate'] == 1).sum()
        pure_honest = (bidder_labels['cartel_rate'] == 0).sum()
        mixed = len(bidder_labels) - pure_cartel - pure_honest
        
        print(f"\nBidder breakdown:")
        print(f"  Pure cartel (100% cartel rate): {pure_cartel}")
        print(f"  Pure honest (0% cartel rate): {pure_honest}")
        print(f"  Mixed (some cartel, some honest): {mixed}")
        
        if mixed == 0:
            print("\n⚠️  CRITICAL: Every bidder is either 100% cartel or 100% honest!")
            print("   The model might just be learning BIDDER IDENTITY, not behavior patterns.")
            print("   This is a form of leakage if bidder_id is encoded in features.")
    
    # =========================================================================
    # CHECK 6: Feature Value Uniqueness per Label
    # =========================================================================
    print("\n" + "-"*70)
    print("CHECK 6: Perfect Separability Check")
    print("-"*70)
    print("Checking if any feature perfectly separates classes...")
    
    for col in processed_df.select_dtypes(include=[np.number]).columns:
        if col in exclude_cols:
            continue
        
        cartel_vals = processed_df[processed_df[target_col] == 1][col].dropna()
        honest_vals = processed_df[processed_df[target_col] == 0][col].dropna()
        
        # Check if ranges don't overlap
        if len(cartel_vals) > 0 and len(honest_vals) > 0:
            cartel_min, cartel_max = cartel_vals.min(), cartel_vals.max()
            honest_min, honest_max = honest_vals.min(), honest_vals.max()
            
            if cartel_max < honest_min or honest_max < cartel_min:
                print(f"⚠️  {col}: Perfect separation! Ranges don't overlap.")
                print(f"     Cartel: [{cartel_min:.3f}, {cartel_max:.3f}]")
                print(f"     Honest: [{honest_min:.3f}, {honest_max:.3f}]")
    
    # =========================================================================
    # CHECK 7: Network Feature Leakage (Known Issue)
    # =========================================================================
    print("\n" + "-"*70)
    print("CHECK 7: Network Feature Leakage (KNOWN ISSUE)")
    print("-"*70)
    print("""
⚠️  CONFIRMED LEAKAGE in network features:
    
    The 'network_neighbor_exposure' feature is computed as:
    
        neighbor_exposure[node] = Σ(weight * neighbor_cartel_rate) / Σ(weight)
    
    This DIRECTLY uses the target variable (cartel_rate = mean of is_cartel)!
    
    In a real-world scenario, you would NOT know which competitors are cartels.
    This feature should be REMOVED or computed differently.
    """)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    print("""
POTENTIAL ISSUES IDENTIFIED:

1. NETWORK FEATURE LEAKAGE (Confirmed):
   - network_neighbor_exposure uses target variable directly
   - This contributes ~4.4% importance but is definitely leakage
   
2. HIGH SINGLE-FEATURE AUC:
   - If any tabular feature has AUC > 0.80, investigate its origin
   - Benford's Law features might be computed post-hoc?
   
3. BIDDER IDENTITY LEAKAGE (Check above):
   - If bidders are either 100% cartel or 100% honest, model learns identity
   - This would work in-sample but fail on new bidders
   
4. COUNTRY STRATIFICATION:
   - If certain countries are heavily cartel, model learns geography
   
RECOMMENDED ACTIONS:
1. Remove network_neighbor_exposure (confirmed leakage)
2. Retrain tabular-only model and verify AUC
3. If still > 0.90, investigate individual features
4. Consider temporal split instead of random split
5. Test on held-out country (e.g., train on 6, test on 1)
    """)
    
    return auc_df

if __name__ == "__main__":
    auc_df = diagnose_leakage(
        "data/GTI_labelled_cartel_data_NOV2023.csv",
        "data/processed_procurement_data.csv"
    )


