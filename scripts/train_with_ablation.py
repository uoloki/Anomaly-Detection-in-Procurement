"""
Model Training with Feature Ablation Study (CORRECTED)

Uses Random Forest (best performing model on bidder-level split)

FIXES APPLIED:
1. BIDDER-LEVEL splitting (not tender) - tests true generalization
2. REMOVED network_neighbor_exposure - it was using target variable (leakage!)
3. Network features now use only structural metrics (no target info)

Remaining network features:
- network_degree: Number of competitors (structural)
- network_clustering: Local network density (structural)
- network_eigenvector: Influence score (structural - but computed without target)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import networkx as nx
from networkx.algorithms import bipartite
import warnings
warnings.filterwarnings('ignore')

import utils

def compute_network_features_no_leakage(raw_data_path):
    """
    Compute network features WITHOUT using target variable.
    
    REMOVED: network_neighbor_exposure (was using cartel_rate = target!)
    KEPT: Structural features only (degree, clustering, eigenvector)
    """
    print("Computing network features (NO LEAKAGE version)...")
    
    df = pd.read_csv(raw_data_path)
    df = df.drop_duplicates().reset_index(drop=True)
    
    graph_df = df[['buyer_id', 'bidder_id', 'tender_id']].dropna()  # Note: NO is_cartel!
    
    # Build bipartite graph (structure only, no target labels)
    B = nx.Graph()
    
    buyers = graph_df['buyer_id'].unique()
    B.add_nodes_from(buyers, bipartite=0, node_type='Buyer')
    
    bidders = graph_df['bidder_id'].unique()
    for bidder in bidders:
        B.add_node(bidder, bipartite=1, node_type='Bidder')
    
    edge_data = graph_df.groupby(['buyer_id', 'bidder_id']).size().reset_index(name='weight')
    for _, row in edge_data.iterrows():
        B.add_edge(row['buyer_id'], row['bidder_id'], weight=row['weight'])
    
    # Project to bidder-bidder network
    P = bipartite.weighted_projected_graph(B, bidders)
    
    # Calculate STRUCTURAL metrics only (no target information)
    clustering_coeff = nx.clustering(P, weight='weight')
    degrees = dict(P.degree())
    
    try:
        eigenvector = nx.eigenvector_centrality(P, weight='weight', max_iter=1000)
    except:
        eigenvector = {n: 0 for n in P.nodes()}
    
    # Create dataframe - NO neighbor_exposure (was leakage!)
    network_features = []
    for bidder in bidders:
        network_features.append({
            'bidder_id': bidder,
            'network_degree': degrees.get(bidder, 0),
            'network_clustering': clustering_coeff.get(bidder, 0),
            'network_eigenvector': eigenvector.get(bidder, 0)
            # REMOVED: network_neighbor_exposure (was using target!)
        })
    
    network_df = pd.DataFrame(network_features)
    print(f"Computed {len(network_df)} bidders with 3 structural features (no leakage)")
    
    return network_df


def build_pipeline(X):
    """Build preprocessing + model pipeline using Random Forest (best performer)"""
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
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    return Pipeline([('preprocessor', preprocessor), ('classifier', model)])


def run_ablation_study(raw_data_path, processed_data_path, output_dir="output/models"):
    utils.setup_plotting_style()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("output/charts", exist_ok=True)
    
    print("="*60)
    print("FEATURE ABLATION STUDY (CORRECTED)")
    print("="*60)
    print("Using BIDDER-LEVEL splitting (true generalization test)")
    print("Network features: Structural only (no target leakage)")
    
    # Load data
    print("\nLoading data...")
    processed_df = pd.read_csv(processed_data_path)
    raw_df = pd.read_csv(raw_data_path).drop_duplicates().reset_index(drop=True)
    
    # Compute network features (no leakage version)
    network_df = compute_network_features_no_leakage(raw_data_path)
    
    # Get bidder_id from raw
    processed_df['bidder_id'] = raw_df['bidder_id'].values
    
    # Merge network features
    merged_df = processed_df.merge(network_df, on='bidder_id', how='left')
    
    # Network feature columns (now only 3, removed neighbor_exposure)
    network_cols = ['network_degree', 'network_clustering', 'network_eigenvector']
    merged_df[network_cols] = merged_df[network_cols].fillna(0)
    
    target_col = 'is_cartel'
    group_col = 'bidder_id'  # CHANGED: Split by bidder!
    
    groups = merged_df[group_col]
    y = merged_df[target_col]
    
    # Define feature sets
    tabular_cols = [c for c in processed_df.columns 
                   if c not in [target_col, 'tender_id', 'bidder_id']]
    
    feature_sets = {
        'Tabular Only': tabular_cols,
        'Network Only': network_cols,
        'Tabular + Network': tabular_cols + network_cols
    }
    
    print(f"\nTabular features: {len(tabular_cols)}")
    print(f"Network features: {len(network_cols)} (structural only)")
    
    # Use GroupKFold with bidder groups
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(merged_df, y, groups))
    trainval_idx, test_idx = splits[0]
    
    results = []
    
    for set_name, feature_cols in feature_sets.items():
        print(f"\n--- {set_name} ({len(feature_cols)} features) ---")
        
        X = merged_df[feature_cols]
        
        X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
        y_trainval, y_test = y.iloc[trainval_idx], y.iloc[test_idx]
        groups_trainval = groups.iloc[trainval_idx]
        
        # Inner CV with bidder groups
        inner_cv = GroupKFold(n_splits=5)
        
        pipeline = build_pipeline(X_trainval)
        
        cv_scores = cross_validate(
            pipeline, X_trainval, y_trainval,
            groups=groups_trainval,
            cv=inner_cv,
            scoring=['roc_auc', 'average_precision'],
            return_train_score=True,
            n_jobs=-1
        )
        
        cv_auc = cv_scores['test_roc_auc'].mean()
        cv_auc_std = cv_scores['test_roc_auc'].std()
        
        # Final test on unseen bidders
        pipeline = build_pipeline(X_trainval)
        pipeline.fit(X_trainval, y_trainval)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        test_auc = roc_auc_score(y_test, y_prob)
        test_ap = average_precision_score(y_test, y_prob)
        
        results.append({
            'Feature Set': set_name,
            'Num Features': len(feature_cols),
            'CV ROC AUC': cv_auc,
            'CV AUC Std': cv_auc_std,
            'Test ROC AUC': test_auc,
            'Test PR AUC': test_ap
        })
        
        print(f"  CV ROC AUC: {cv_auc:.4f} ± {cv_auc_std:.4f}")
        print(f"  Test ROC AUC (unseen bidders): {test_auc:.4f}")
        print(f"  Test PR AUC: {test_ap:.4f}")
    
    # Results summary
    results_df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("ABLATION RESULTS (BIDDER-LEVEL SPLIT) - Random Forest")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    
    # Generate ROC/PR curves for best model (Tabular + Network)
    print("\nGenerating ROC/PR curves for Random Forest (Tabular + Network)...")
    X_combined = merged_df[tabular_cols + network_cols]
    X_trainval_comb, X_test_comb = X_combined.iloc[trainval_idx], X_combined.iloc[test_idx]
    y_trainval_comb, y_test_comb = y.iloc[trainval_idx], y.iloc[test_idx]
    
    pipeline_roc = build_pipeline(X_trainval_comb)
    pipeline_roc.fit(X_trainval_comb, y_trainval_comb)
    y_prob_roc = pipeline_roc.predict_proba(X_test_comb)[:, 1]
    
    test_auc_roc = roc_auc_score(y_test_comb, y_prob_roc)
    test_ap_roc = average_precision_score(y_test_comb, y_prob_roc)
    
    fpr, tpr, _ = roc_curve(y_test_comb, y_prob_roc)
    precision, recall, _ = precision_recall_curve(y_test_comb, y_prob_roc)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'Random Forest (AUC={test_auc_roc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve - Random Forest (Tabular + Network)\nTested on Unseen Bidders')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(recall, precision, 'r-', linewidth=2, label=f'Random Forest (AP={test_ap_roc:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve - Random Forest (Tabular + Network)\nTested on Unseen Bidders')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    utils.save_plot(fig, "roc_pr_curves.png")
    plt.close()
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, results_df['CV ROC AUC'], width, 
                   yerr=results_df['CV AUC Std'], capsize=5,
                   label='CV ROC AUC', color='steelblue')
    bars2 = ax.bar(x + width/2, results_df['Test ROC AUC'], width,
                   label='Test ROC AUC', color='coral')
    
    ax.set_ylabel('ROC AUC')
    ax.set_title('Feature Ablation Study (Bidder-Level Split)\nTested on UNSEEN Bidders')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Feature Set'])
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    utils.save_plot(fig, "ablation_study.png")
    plt.close()
    
    # Feature importance for combined model
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Tabular + Network)")
    print(f"{'='*60}")
    
    X_combined = merged_df[tabular_cols + network_cols]
    X_trainval_comb = X_combined.iloc[trainval_idx]
    y_trainval_comb = y.iloc[trainval_idx]
    
    # Use Random Forest for importance
    numeric_cols_comb = X_combined.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols_comb = X_combined.select_dtypes(include=['object', 'category']).columns.tolist()
    
    transformers_comb = []
    if numeric_cols_comb:
        transformers_comb.append(('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_cols_comb))
    
    if categorical_cols_comb:
        transformers_comb.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols_comb))
    
    preprocessor_comb = ColumnTransformer(transformers=transformers_comb)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    pipeline_rf = Pipeline([
        ('preprocessor', preprocessor_comb),
        ('classifier', rf_model)
    ])
    
    pipeline_rf.fit(X_trainval_comb, y_trainval_comb)
    
    # Get feature names
    feature_names = numeric_cols_comb.copy()
    if categorical_cols_comb:
        ohe = pipeline_rf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        ohe_names = ohe.get_feature_names_out(categorical_cols_comb)
        feature_names.extend(ohe_names)
    
    importances = pipeline_rf.named_steps['classifier'].feature_importances_
    
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    feat_imp['Type'] = feat_imp['Feature'].apply(
        lambda x: 'Network' if x.startswith('network_') else 'Tabular'
    )
    
    print("\nTop 20 Features:")
    print(feat_imp.head(20).to_string(index=False))
    
    # Plot feature importance
    top_n = 25
    top_feat = feat_imp.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['coral' if t == 'Network' else 'steelblue' for t in top_feat['Type']]
    
    ax.barh(range(len(top_feat)), top_feat['Importance'].values, color=colors)
    ax.set_yticks(range(len(top_feat)))
    ax.set_yticklabels(top_feat['Feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances\n(Blue=Tabular, Red=Network, No Leakage)')
    
    plt.tight_layout()
    utils.save_plot(fig, "feature_importance_with_network.png")
    plt.close()
    
    # Network features summary
    network_importance = feat_imp[feat_imp['Type'] == 'Network']['Importance'].sum()
    tabular_importance = feat_imp[feat_imp['Type'] == 'Tabular']['Importance'].sum()
    total = network_importance + tabular_importance
    
    print(f"\n--- Feature Type Contribution ---")
    print(f"Tabular features: {tabular_importance:.4f} ({tabular_importance/total*100:.1f}%)")
    print(f"Network features: {network_importance:.4f} ({network_importance/total*100:.1f}%)")
    
    # Save results
    results_df.to_csv(f"{output_dir}/ablation_results.csv", index=False)
    feat_imp.to_csv(f"{output_dir}/feature_importance_combined.csv", index=False)
    
    print(f"\n✓ Results saved to {output_dir}/")
    print(f"✓ Charts saved to output/charts/")
    print(f"\n⚠️  NOTE: Results now reflect TRUE generalization (bidder-level split)")
    print(f"   Network features no longer have target leakage")
    
    return results_df, feat_imp

if __name__ == "__main__":
    run_ablation_study(
        raw_data_path="data/GTI_labelled_cartel_data_NOV2023.csv",
        processed_data_path="data/processed_procurement_data.csv"
    )
