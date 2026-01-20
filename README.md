# Cartel Detection in Public Procurement

## A Machine Learning and Network Analysis Approach

---

## Executive Summary

This project develops a machine learning system to detect cartel behavior in public procurement data using rigorous validation methodology that tests generalization to unseen bidders.

**Key Results:**

| Metric | Value |
|--------|-------|
| Best Model | Random Forest (Tabular + Network Features) |
| Test ROC AUC | 89.0% |
| Test PR AUC | 85.3% |
| Validation Method | Bidder-level split (unseen bidders) |

**Key Findings:**

- Benford's Law deviation is the strongest predictor of cartel behavior
- Network structural features provide marginal improvement (+0.3% AUC)
- Model generalizes to unseen bidders with 89.0% AUC
- Results align with academic benchmarks (Fazekas et al.: 77-91% accuracy)

---

## 1. Introduction and Literature Review

### 1.1 Background

Public procurement accounts for approximately 12% of global GDP (around $11 trillion USD annually). Bid-rigging cartels impose significant costs on public budgets, with overcharges estimated at 10-50% above competitive prices.

### 1.2 Data Source

This project uses the GTI Labelled Cartel Dataset developed by the Government Transparency Institute:

> **Citation**: Fazekas, M., Toth, B., and Wachs, J. (2023). *Public procurement cartels: A large-sample testing of screens using machine learning*. Government Transparency Institute Working Paper GTI-WP/2023:02.

The dataset contains:

- 78 proven cartel cases from 7 European countries
- Countries covered: Bulgaria, France, Hungary, Latvia, Portugal, Spain, Sweden
- Time period: 2007-2020
- Labeling methodology: Judicial records matched with procurement data

### 1.3 Prior Work and Our Contribution

The original Fazekas et al. (2023) paper achieved 77-91% prediction accuracy using Random Forest with contract-level cartel screens.

**Our Contributions:**

1. Bidder-Level Validation: Tested generalization to truly unseen bidders
2. Network Feature Investigation: Systematically tested graph-based features (negative result)
3. Reproducible Pipeline: Clean, documented codebase

### 1.4 Performance Comparison

| Metric | Fazekas et al. (2023) | Our Model |
|--------|------------------------|-----------|
| Algorithm | Random Forest | Random Forest |
| Performance | 77-91% accuracy | 89.0% ROC AUC |
| Validation | Not specified | Bidder-level split |

---

## 2. Data Description

### 2.1 Dataset Overview

| Metric | Value |
|--------|-------|
| Source | Government Transparency Institute (GTI) |
| Original Size | 15,616 rows x 34 columns |
| After Deduplication | 12,870 rows x 34 columns |
| Unique Tenders | 10,565 |
| Unique Bidders | 1,001 |

### 2.2 Target Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Non-Cartel (0) | 7,071 | 54.9% |
| Cartel (1) | 5,799 | 45.1% |

### 2.3 Bidder Behavior Analysis

A critical finding for methodology:

| Bidder Type | Count | Percentage |
|-------------|-------|------------|
| Always Cartel (100% cartel rate) | 503 | 50.2% |
| Always Honest (0% cartel rate) | 311 | 31.1% |
| Mixed behavior | 187 | 18.7% |

Most bidders (81%) are consistently cartel or honest across all their bids, which informed our validation strategy of splitting by bidder rather than by tender.

---

## 3. Methodology

### 3.1 Validation Strategy: Bidder-Level Splitting

Since most bidders are consistently cartel or honest, we use bidder-level splitting to test true generalization:

```
Train: Bidder_A (all tenders), Bidder_B (all tenders)
Test:  Bidder_C (all tenders), Bidder_D (all tenders)
       Model has never seen these bidders
```

This tests the real-world question: Can we detect new cartel members we have never seen before?

### 3.2 Cross-Validation Setup

| Component | Configuration |
|-----------|---------------|
| Outer Split | 80% train, 20% test (by bidder) |
| Inner CV | 5-fold GroupKFold (by bidder) |
| Stratification | Maintained class balance |

### 3.3 Generalization Tests

| Split Method | ROC AUC | Interpretation |
|--------------|---------|----------------|
| By Bidder | 0.890 | Generalization to new bidders |
| Leave-Country-Out | 0.575 | Poor cross-border transfer |

---

## 4. Exploratory Data Analysis

### 4.1 PCA Visualization

![PCA Projection](../output/charts/pca_projection.png)

The first two principal components capture only approximately 30% of total variance. This indicates:

1. Cartel detection is inherently high-dimensional
2. No simple 2D projection separates cartels from honest bidders
3. Ensemble methods are appropriate for this problem

### 4.2 Feature Distributions

![Key Numeric Features](../output/charts/key_numeric_features.png)

---

## 5. Network Analysis (Negative Result)

### 5.1 Graph Construction

We constructed a Bidder-Bidder Competitor Network:

- Nodes: 2,763 bidders
- Edges: 44,665 competitor relationships  
- Edge Definition: Two bidders connected if they bid for the same buyer
- Weight: Number of shared buyers/tenders

![Network Topology](../output/charts/network_topology.png)

### 5.2 Network Features Computed

| Feature | Description |
|---------|-------------|
| network_degree | Number of competitors |
| network_clustering | Local network density |
| network_eigenvector | Influence/centrality score |

### 5.3 Ablation Study Results (Random Forest)

| Feature Set | Test ROC AUC | Test PR AUC | Interpretation |
|-------------|--------------|-------------|----------------|
| Tabular Only | 0.887 | 0.845 | Strong baseline |
| Network Only | 0.589 | 0.497 | Weak on its own |
| Tabular + Network | 0.890 | 0.853 | Best performance |

![Ablation Study](../output/charts/ablation_study.png)

### 5.4 Conclusion: Network Features Provide Marginal Improvement

Network structural features alone are weak predictors (58.9% AUC), but when combined with tabular features, they provide a small improvement:

- Tabular Only: 88.7% AUC
- Tabular + Network: 89.0% AUC (+0.3%)

The network features contribute about 11% of total feature importance, with clustering coefficient and degree being the most useful structural metrics.

---

## 6. Model Performance

### 6.1 Model Comparison (Bidder-Level Split)

| Model | Test ROC AUC | Test PR AUC |
|-------|--------------|-------------|
| Logistic Regression | 0.553 | - |
| Random Forest | 0.890 | 0.853 |
| Gradient Boosting | 0.835 | 0.817 |

Note: Random Forest results shown with Tabular + Network features.

![Model Comparison](../output/charts/model_comparison.png)

### 6.2 ROC and Precision-Recall Curves

![ROC and PR Curves](../output/charts/roc_pr_curves.png)

---

## 7. Feature Importance

### 7.1 Top Predictive Features

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | benfords_market_yearly_avg | 9.8% | Price digit manipulation |
| 2 | benfords_market_yearly_avg_lag | 9.1% | Historical price patterns |
| 3 | lot_bidscount_avg | 7.3% | Competition manipulation |
| 4 | lot_bidscount_avg_lag | 6.1% | Historical bid patterns |
| 5 | contract_count_bidder_yearly | 4.3% | Bidder activity level |

![Feature Importance](../output/charts/feature_importance_with_network.png)

### 7.2 Feature Categories

| Category | Total Importance | Key Insight |
|----------|------------------|-------------|
| Benford's Law | ~20% | Rigged prices do not follow natural digit patterns |
| Bid Counts | ~15% | Cartels manipulate apparent competition |
| Market Structure | ~10% | Bidder-buyer relationship patterns |
| Network (structural) | ~11% | Provides marginal improvement |

---

## 8. Conclusions

### 8.1 Key Findings

1. Benford's Law is the strongest predictor: Forensic accounting features outperform all others
2. Network features provide marginal benefit: +0.3% AUC when combined with tabular features
3. 89.0% AUC on unseen bidders: Aligns with academic benchmarks (77-91%)
4. Cross-border generalization is poor: Country-specific patterns limit transfer

### 8.2 Final Model Summary

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest |
| Features | 28 (25 tabular + 3 network) |
| Test ROC AUC | 89.0% |
| Test PR AUC | 85.3% |
| Validation | Bidder-level split |
| Generalization | Tested on unseen bidders |

### 8.3 Practical Recommendations

1. Risk Scoring: Flag bids with probability above 0.5 for manual review
2. Focus on Behavior: Monitor Benford's Law deviations and bid count anomalies
3. New Bidder Detection: Model generalizes to previously unseen bidders
4. Cross-Border Limitation: Model does not transfer well across countries (0.58 AUC)

### 8.4 Limitations

- Model trained on historical cases from 7 EU countries
- Poor cross-border generalization (country-specific patterns)
- May miss novel cartel tactics not in training data
- Network features alone are weak; only useful combined with behavioral features

---

## References

1. Fazekas, M., Toth, B., and Wachs, J. (2023). *Public procurement cartels: A large-sample testing of screens using machine learning*. Government Transparency Institute Working Paper GTI-WP/2023:02.

2. Bosio, E., et al. (2020). Public Procurement in Law and Practice. *American Economic Review*.

3. Porter, R. H., and Zona, J. D. (1999). Ohio school milk markets: An analysis of bidding. *RAND Journal of Economics*.

---

## Appendix A: Reproducibility

### Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Process data
python scripts/process_data.py

# 2. Exploratory Data Analysis
python scripts/eda.py

# 3. Train model (bidder-level split)
python scripts/train_model.py

# 4. Graph analysis
python scripts/graph_analysis.py

# 5. Ablation study
python scripts/train_with_ablation.py

# 6. Validation checks
python scripts/validate_leakage.py
python scripts/test_generalization.py
```

### Output Files

| Type | Location |
|------|----------|
| Processed Data | data/processed_procurement_data.csv |
| Charts | output/charts/ |
| Model Metrics | output/models/ |

---

## Appendix B: Dataset Selection Rationale

### Initial Dataset: FOPPA 1.1.3

The project initially planned to use the FOPPA Dataset v1.1.3, which contains French public procurement data.

Problems Identified:

1. No Ground Truth Labels: FOPPA lacks explicit cartel/collusion labels
2. Only Weak Supervision Possible: Would require proxy labels or unsupervised methods
3. Limited Validation: No way to verify detected anomalies are actual cartels

### Switch to GTI Dataset

We switched to the GTI Labelled Cartel Dataset because:

1. Confirmed Labels: 78 proven cartel cases from judicial records
2. Multi-Country Coverage: 7 European countries
3. Academic Validation: Peer-reviewed research benchmark
4. Rich Features: Pre-computed forensic indicators

This enabled rigorous supervised learning with proper evaluation.

