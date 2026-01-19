"""
Model Training Script with BIDDER-LEVEL Splitting

CRITICAL INSIGHT: The model was memorizing bidder identities, not learning behavior.
- Tender-level split: 0.986 AUC (inflated - same bidders in train/test)
- Bidder-level split: 0.814 AUC (realistic - unseen bidders in test)

Solution: Use GroupKFold with bidder_id as group variable.
This tests: "Can we detect NEW cartel members we haven't seen before?"
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score, 
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve
)
import utils

def verify_no_bidder_leakage(train_bidders, test_bidders):
    """
    Sanity check: Ensure no bidder_id appears in multiple sets.
    """
    train_set = set(train_bidders)
    test_set = set(test_bidders)
    
    overlap = train_set.intersection(test_set)
    
    if overlap:
        raise ValueError(f"LEAKAGE DETECTED! {len(overlap)} bidders in both train and test!")
    
    print("✓ No bidder leakage detected between sets")
    return True

def train_model(processed_path, raw_path, output_dir="output/models"):
    utils.setup_plotting_style()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("output/charts", exist_ok=True)
    
    # Load data
    print(f"Loading data...")
    processed_df = pd.read_csv(processed_path)
    raw_df = pd.read_csv(raw_path).drop_duplicates().reset_index(drop=True)
    
    # Get bidder_id from raw (aligned after dedup)
    processed_df['bidder_id'] = raw_df['bidder_id'].values
    
    target_col = 'is_cartel'
    group_col = 'bidder_id'  # CHANGED: Split by bidder, not tender
    
    # Separate Groups and Features
    groups = processed_df[group_col]
    y = processed_df[target_col]
    X = processed_df.drop(columns=[target_col, 'tender_id', group_col], errors='ignore')
    
    print(f"\n{'='*50}")
    print("DATA OVERVIEW")
    print(f"{'='*50}")
    print(f"Total Samples: {len(X)}")
    print(f"Total Unique Bidders: {groups.nunique()}")
    print(f"Features: {X.shape[1]}")
    print(f"\nTarget Distribution:")
    print(y.value_counts(normalize=True).to_string())
    
    # Feature Types
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\nNumeric Features: {len(numeric_cols)}")
    print(f"Categorical Features: {len(categorical_cols)}")
    
    # Pipeline Setup
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # =========================================================================
    # SPLITTING STRATEGY: BIDDER-LEVEL Split
    # =========================================================================
    # This ensures the test set contains ONLY bidders the model has never seen.
    # Tests true generalization: "Can we detect NEW cartel members?"
    # =========================================================================
    
    print(f"\n{'='*50}")
    print("SPLITTING STRATEGY: BIDDER-LEVEL")
    print(f"{'='*50}")
    print("Testing on UNSEEN bidders (true generalization test)")
    
    # Use GroupKFold to split by bidder
    gkf = GroupKFold(n_splits=5)
    
    # Get first split for train/test
    splits = list(gkf.split(X, y, groups))
    trainval_idx, test_idx = splits[0]
    
    X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
    y_trainval, y_test = y.iloc[trainval_idx], y.iloc[test_idx]
    groups_trainval, groups_test = groups.iloc[trainval_idx], groups.iloc[test_idx]
    
    # SANITY CHECK: Verify no leakage
    verify_no_bidder_leakage(groups_trainval, groups_test)
    
    print(f"\nTrain+Val Set: {len(X_trainval)} samples, {groups_trainval.nunique()} unique bidders")
    print(f"Test Set: {len(X_test)} samples, {groups_test.nunique()} unique bidders")
    print(f"Train cartel rate: {y_trainval.mean():.1%}")
    print(f"Test cartel rate: {y_test.mean():.1%}")
    
    # Define Models
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced', 
            max_iter=1000, 
            random_state=42,
            solver='lbfgs'
        ),
        "Random Forest": RandomForestClassifier(
            class_weight='balanced', 
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            max_depth=10,
            min_samples_leaf=5
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        )
    }
    
    # =========================================================================
    # CROSS-VALIDATION (Bidder-Level)
    # =========================================================================
    print(f"\n{'='*50}")
    print("CROSS-VALIDATION (5-Fold, Bidder-Level)")
    print(f"{'='*50}")
    
    inner_cv = GroupKFold(n_splits=5)
    
    cv_results = []
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Cross-validate with bidder groups
        scores = cross_validate(
            pipeline, 
            X_trainval, 
            y_trainval,
            groups=groups_trainval,
            cv=inner_cv,
            scoring=['roc_auc', 'average_precision', 'f1'],
            return_train_score=True,
            n_jobs=-1
        )
        
        cv_result = {
            'Model': name,
            'CV ROC AUC (mean)': scores['test_roc_auc'].mean(),
            'CV ROC AUC (std)': scores['test_roc_auc'].std(),
            'CV PR AUC (mean)': scores['test_average_precision'].mean(),
            'CV F1 (mean)': scores['test_f1'].mean(),
            'Train ROC AUC (mean)': scores['train_roc_auc'].mean(),
            'Overfit Gap': scores['train_roc_auc'].mean() - scores['test_roc_auc'].mean()
        }
        cv_results.append(cv_result)
        
        print(f"  CV ROC AUC: {cv_result['CV ROC AUC (mean)']:.4f} ± {cv_result['CV ROC AUC (std)']:.4f}")
        print(f"  Train ROC AUC: {cv_result['Train ROC AUC (mean)']:.4f}")
        print(f"  Overfit Gap: {cv_result['Overfit Gap']:.4f}")
    
    cv_df = pd.DataFrame(cv_results)
    print(f"\n{cv_df.to_string(index=False)}")
    
    best_model_name = cv_df.loc[cv_df['CV ROC AUC (mean)'].idxmax(), 'Model']
    print(f"\n★ Best Model (by CV ROC AUC): {best_model_name}")
    
    # =========================================================================
    # FINAL EVALUATION ON HELD-OUT TEST SET (UNSEEN BIDDERS)
    # =========================================================================
    print(f"\n{'='*50}")
    print("FINAL TEST SET EVALUATION (UNSEEN BIDDERS)")
    print(f"{'='*50}")
    
    final_results = []
    
    for name, model in models.items():
        print(f"\nTraining final {name} on Train+Val set...")
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_trainval, y_trainval)
        
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        
        final_results.append({
            'Model': name,
            'Test ROC AUC': auc,
            'Test PR AUC': ap
        })
        
        print(f"  Test ROC AUC: {auc:.4f}")
        print(f"  Test PR AUC: {ap:.4f}")
        
        # Save metrics
        report = classification_report(y_test, y_pred)
        with open(f"{output_dir}/metrics_{name.replace(' ', '_')}.txt", "w") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Split: BIDDER-LEVEL (unseen bidders in test)\n")
            f.write(f"{'='*40}\n\n")
            f.write(f"Test ROC AUC: {auc:.4f}\n")
            f.write(f"Test PR AUC: {ap:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues', normalize='true', ax=ax)
        plt.title(f"Confusion Matrix - {name}\n(Tested on Unseen Bidders)")
        plt.grid(False)
        utils.save_plot(fig, f"confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close()
        
        # ROC/PR Curves for best model
        if name == best_model_name:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'{name} (AUC={auc:.3f})')
            axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0].set_xlabel('False Positive Rate')
            axes[0].set_ylabel('True Positive Rate')
            axes[0].set_title('ROC Curve (Unseen Bidders)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(recall, precision, 'r-', linewidth=2, label=f'{name} (AP={ap:.3f})')
            axes[1].set_xlabel('Recall')
            axes[1].set_ylabel('Precision')
            axes[1].set_title('Precision-Recall Curve (Unseen Bidders)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            utils.save_plot(fig, "roc_pr_curves.png")
            plt.close()
    
    final_df = pd.DataFrame(final_results)
    print(f"\n{'='*50}")
    print("MODEL COMPARISON (Test Set - Unseen Bidders)")
    print(f"{'='*50}")
    print(final_df.to_string(index=False))
    
    # Model comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(final_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, final_df['Test ROC AUC'], width, label='ROC AUC', color='steelblue')
    bars2 = ax.bar(x + width/2, final_df['Test PR AUC'], width, label='PR AUC', color='coral')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison - Tested on UNSEEN Bidders\n(True Generalization Performance)')
    ax.set_xticks(x)
    ax.set_xticklabels(final_df['Model'])
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    utils.save_plot(fig, "model_comparison.png")
    plt.close()
    
    # Save results
    cv_df.to_csv(f"{output_dir}/cv_results.csv", index=False)
    final_df.to_csv(f"{output_dir}/test_results.csv", index=False)
    
    print(f"\n✓ All results saved to {output_dir}/")
    print(f"✓ Charts saved to output/charts/")
    print(f"\n⚠️  NOTE: These results reflect TRUE generalization to unseen bidders.")
    print(f"   Expected AUC range: 0.75-0.85 (realistic for cartel detection)")

if __name__ == "__main__":
    train_model(
        processed_path="data/processed_procurement_data.csv",
        raw_path="data/GTI_labelled_cartel_data_NOV2023.csv"
    )
