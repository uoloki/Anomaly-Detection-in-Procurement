# Data Dictionary

## Overview

This dataset contains procurement tender data labelled with cartel activity, sourced from the Government Transparency Institute (GTI).

**Citation**: Fazekas, M., Toth, B., and Wachs, J. (2023). *Public procurement cartels: A large-sample testing of screens using machine learning*. Government Transparency Institute Working Paper GTI-WP/2023:02.

**Coverage**: 78 proven cartels from 7 European countries (Bulgaria, France, Hungary, Latvia, Portugal, Spain, Sweden), 2007-2020.

---

## Key Identifiers

| Column | Description | Notes |
|--------|-------------|-------|
| tender_id | Unique identifier for the tender/contract | Dropped during training |
| buyer_id | Unique identifier for the buying organization | Dropped during training (leakage risk) |
| bidder_id | Unique identifier for the bidding company | Used for train/test split (unseen bidders in test) |
| lot_id | Unique identifier for the lot within a tender | Dropped during training |
| cartel_id | Identifier linking to proven cartel case | Dropped during training (direct leak) |

---

## Target Variable

| Column | Type | Description |
|--------|------|-------------|
| is_cartel | Binary (0/1) | Target: Whether the bid is part of a confirmed cartel case |

---

## Features by Category

### Price-Based Screens (Most Important)

| Column | Paper Terminology | Type | Description | Importance |
|--------|-------------------|------|-------------|------------|
| benfords_market_yearly_avg | Price-based forensic screen | Float | Deviation from Benford's Law for the market/year | Rank 1 (9.8%) |
| relative_value | Relative price | Float | Bid price relative to estimate | Medium |

**Rationale** (from Fazekas et al.): Cartels cannot perfectly randomize rigged bid prices. They leave statistical artifacts detectable through forensic accounting methods like Benford's Law analysis.

### Bidding Pattern Screens

| Column | Paper Terminology | Type | Description | Importance |
|--------|-------------------|------|-------------|------------|
| lot_bidscount | Number of bidders | Integer | Number of bids received for the lot | Rank 3 (7.3%) |
| singleb_avg | Single bidding rate | Float | Market-level average of single-bidder tenders | Medium |
| bid_isconsortium | Consortia indicator | Binary | Whether the bid was from a consortium | Low |
| bid_issubcontracted | Subcontracting indicator | Binary | Whether the bid involved subcontracting | Medium |

**Rationale**: Cartels manipulate the appearance of competition. They may use cover bids (fake losing bids) to avoid single-bidder suspicion.

### Market Structure Screens

| Column | Paper Terminology | Type | Description |
|--------|-------------------|------|-------------|
| buyer_avg_bidder_yearly | Buyer activity | Float | Average number of bidders this buyer attracts per year |
| contract_count_bidder_yearly | Bidder activity | Float | Number of contracts won/bid by the bidder in that year |

**Rationale**: Cartels often exhibit shell company profiles with low contract counts, targeting specific high-volume buyers.

---

## Leakage Columns (Removed During Processing)

| Column | Reason for Removal |
|--------|-------------------|
| cartel_tender | Direct label leak, perfectly correlated with target |
| persistent_id | High-cardinality identifier |
| tender_id | Identifier (not a feature) |
| lot_id | High-cardinality identifier |
| buyer_id | Enables memorization of buyer-specific patterns |
| bidder_id | Used for splitting only, not as a feature |
| cartel_id | Direct reference to cartel case |
| tender_year | Temporal metadata |

---

## Graph/Network Features (Not Used: Negative Result)

Network features were investigated but found to be not predictive when properly validated.

### Features Computed (for analysis only)

| Feature | Type | Test AUC | Description |
|---------|------|----------|-------------|
| network_degree | Float | ~0.51 | Number of competitor connections |
| network_clustering | Float | ~0.51 | Local density of bidder's ego-network |
| network_eigenvector | Float | ~0.51 | Bidder's influence in competitor network |

### Removed Feature (Data Leakage)

| Feature | Issue |
|---------|-------|
| network_neighbor_exposure | Removed: Was computed using target variable (cartel_rate of neighbors) |

### Network Definition

- Nodes: 2,763 bidders
- Edges: 44,665 competitor relationships
- Edge Rule: Two bidders connected if they bid for the same buyer
- Weight: Number of shared buyers/tenders

### Ablation Study Results

| Feature Set | Test ROC AUC |
|-------------|--------------|
| Tabular Only | 0.835 (Best) |
| Network Only | 0.507 (Random) |
| Tabular + Network | 0.796 (Worse) |

**Conclusion**: Structural network features add noise and hurt performance. Only tabular behavioral features are predictive.

---

## Data Quality Notes

1. Duplicates: 2,746 duplicate rows removed during processing
2. Missing Values: Handled via median imputation (numeric) or constant fill (categorical)
3. Class Balance: Approximately 45% cartel cases (pre-filtered high-risk subset)

---

## Validation Methodology

### Bidder-Level Splitting (Correct)

```
Train Set: Bidders A, B, C (all their tenders)
Test Set:  Bidders D, E, F (all their tenders) - Never seen
```

This tests true generalization: Can we detect new cartel members?

### Why Tender-Level Split Was Incorrect

Approximately 70% of bidders are consistently cartel or honest across all their bids:

- 503 bidders: 100% cartel
- 33 bidders: 0% cartel

With tender-level split, the model memorized bidder identities, leading to inflated AUC (0.99 vs true 0.83).

---

## Feature Availability by Country

Based on Fazekas et al. (2023), Table 7.2:

| Feature Type | BG | FR | HU | LV | PT | ES | SE |
|--------------|----|----|----|----|----|----|-----|
| Relative price | Yes | No | Yes | Yes | Yes | Yes | No |
| Single bidding | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Number of bidders | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Consortia | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Subcontracting | No | Yes | No | Yes | No | Yes | No |

---

*Last updated: January 2026*
