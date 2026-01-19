# Project Structure

This project detects cartel behavior in procurement data using machine learning. Through rigorous validation, we achieved 83.5% ROC AUC on unseen bidders.

## Directory Layout

### data/

Contains the raw and processed datasets.

- GTI_labelled_cartel_data_NOV2023.csv: Original dataset from GTI
- processed_procurement_data.csv: Cleaned dataset (leakage columns removed)

### scripts/

Modular Python scripts for the pipeline.

| Script | Purpose |
|--------|---------|
| utils.py | Shared helper functions (data loading, plotting) |
| process_data.py | Data cleaning, deduplication, leakage removal |
| eda.py | Exploratory data analysis and charts |
| train_model.py | Main model training (bidder-level split) |
| train_with_ablation.py | Feature ablation study (tabular vs network) |
| graph_analysis.py | Network construction and visualization |
| validate_leakage.py | Data integrity checks |
| diagnose_leakage.py | Comprehensive leakage investigation |
| test_generalization.py | Generalization tests (bidder/country splits) |

### notebooks/

Interactive notebooks for exploration.

- eda_workflow.ipynb: Original EDA notebook
- graph_analysis.ipynb: Network analysis notebook (if present)

### output/

Generated artifacts.

- charts/: Visualizations (ROC curves, feature importance, etc.)
- models/: Performance metrics, classification reports

### docs/

Project documentation.

- PROJECT_REPORT.md: Full project writeup with findings
- DATA_DICTIONARY.md: Dataset column explanations
- PROJECT_STRUCTURE.md: This file

---

## Pipeline Workflow

```
Data Pipeline
--------------

Raw Data
   |
   v
process_data.py --> Cleaned Data
   |                    |
   |                    |--> eda.py --> Charts
   |                    |
   |                    |--> train_model.py --> Final Model
   |                    |    (bidder-level split)
   |                    |
   |                    |--> graph_analysis.py --> Network Insights
   v
Validation
   |--> validate_leakage.py (integrity checks)
   |--> diagnose_leakage.py (leakage investigation)
   |--> test_generalization.py (bidder/country holdout)
```

---

## Key Methodology Decisions

### 1. Bidder-Level Splitting

Train/test split by bidder_id, not tender_id. This ensures the test set contains only bidders the model has never seen.

### 2. Leakage Removal

- Removed ID columns that enable memorization
- Removed network_neighbor_exposure (used target variable)
- Verified no single feature has AUC above 0.70

### 3. Network Features (Negative Result)

Network structural features were investigated but found to hurt performance. Only tabular behavioral features are used in the final model.

---

## Running the Pipeline

```bash
# 1. Clean and process data
python scripts/process_data.py

# 2. Generate EDA charts
python scripts/eda.py

# 3. Train final model (bidder-level split)
python scripts/train_model.py

# 4. (Optional) Network analysis
python scripts/graph_analysis.py

# 5. (Optional) Ablation study
python scripts/train_with_ablation.py

# 6. (Optional) Leakage validation
python scripts/validate_leakage.py
python scripts/diagnose_leakage.py
python scripts/test_generalization.py
```

---

## Final Results

| Metric | Value |
|--------|-------|
| Best Model | Gradient Boosting |
| Features | 25 tabular (behavioral) |
| Test ROC AUC | 83.5% |
| Validation | Bidder-level split |

See docs/PROJECT_REPORT.md for full analysis.
