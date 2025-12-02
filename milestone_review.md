# Milestone Review: Corruption and Fraud Detection in Public Procurement

## 1. Dataset Description and Analysis

### Dataset
We use the FOPPA French Public Procurement dataset (v1.1.3), which contains:

- **Agents.csv**: contracting authorities and suppliers  
- **Lots.csv**: individual tender lots with metadata  
- **LotBuyers.csv / LotSuppliers.csv**: buyer–supplier award relationships  
- **Criteria.csv**: evaluation criteria information  

The dataset covers the period **2010–2020** and includes information such as award prices, award dates, procedure types, number of bidders, CPV codes, and supplier identities.

### Data Characteristics
- Highly heterogeneous, covering various market sectors and administrative regions.  
- Large-scale bipartite structure: buyers and suppliers connected through awarded lots.  
- Strong long-tail distributions:  
  - Many suppliers appear only once.  
  - Many buyers award only a small number of lots.  
- Substantial missing or inconsistent fields:
  - Approximately **21 percent** of award prices missing or zero.
  - Approximately **13 percent** of bidder counts missing.
  - Zero-price or extremely small-price lots are common and required filtering.

### Many CPV Sectors have placeholders for price -> requires imputation or careful dropping
<img width="626" height="393" alt="image" src="https://github.com/user-attachments/assets/7232e1f9-bbbd-484d-bedd-def6c131db54" />

### Estimated and award price are heavily correlated, but still vary
<img width="602" height="547" alt="image" src="https://github.com/user-attachments/assets/be2cc19c-4fa5-4578-a0c2-5823a3b33aa1" />

### For some CPV sectors, the award price is lower than estimated
<img width="1176" height="547" alt="image" src="https://github.com/user-attachments/assets/6e572137-a980-4e16-a9d5-815011c3b43e" />


### Data Cleaning and Preparation
We conducted the following steps:

- Converted and validated numeric fields (award price, estimated price, tender counts).  
- Removed invalid dates and filtered the timespan to 2010–2020.  
- Dropped corrupted rows with non-positive award prices.  
- Binned numeric features into categories for pattern mining:  
  - `price_bin`, `nt_bin` (number of tenders), CPV2 code, procedure type.  
- Constructed a cleaned mining subset containing:
  - **9,499 buyers**
  - **206,364 buyer–supplier edges**
  - Unified edge labels of the form:  
    `CPV2 code | price quantile | number of bidders | tender type`

This cleaned subset is suitable for graph-based anomaly detection and pattern mining.


## 2. Machine Learning Models Selected

### Graph-Based Anomaly Detection
Our approach follows the methodology used in recent procurement anomaly detection literature:

1. **Graph Construction**  
   Each buyer is represented as a star-shaped bipartite graph linking the buyer node to supplier nodes, with edges labeled by relevant tender metadata.

2. **Weak Supervision for Labels**  
   We assigned heuristic labels to buyers based on known risk indicators:
   - Single-bidding rate
   - Supplier concentration
   - CPV concentration (Herfindahl index)

   Buyers in the top decile of these scores were labeled as anomalous, and buyers with consistently low-risk indicators were labeled as normal.

3. **Pattern Mining Methods**
   - **PrefixSpan** for frequent sequence mining on edge labels.  
   - Planned extensions include:
     - frequent itemset mining (FP-growth)  
     - subgraph pattern mining (gSpan)** on smaller graph subsets due to computational costs.

4. **Baseline Anomaly Metrics**
   A combined anomaly score was generated through normalized measures of:
   - single-bidding dominance  
   - top supplier share  
   - CPV concentration  

This score was used to identify buyers likely to exhibit suspicious procurement behavior.

## 3. Preliminary Results

### Descriptive Findings
Initial exploratory analysis reveals the following:

- Number of bidders is often low; single-bid lots are frequent across sectors.
<img width="1387" height="590" alt="image" src="https://github.com/user-attachments/assets/07585692-df21-4d1d-8add-23d6439dd6ed" />

- CPV distributions highlight a concentration of procurement activity in a small number of categories (e.g., CPV 45, 66, 90).

### Graph-Level Observations
- Buyer graphs typically connect to **3–15 suppliers**, though some large authorities connect to hundreds.  
- Supplier degree distribution shows extreme skew, consistent with procurement markets where many suppliers bid rarely.

<img width="619" height="642" alt="image" src="https://github.com/user-attachments/assets/4299c9d4-ddc0-4c21-a911-a7728e675695" />
<img width="619" height="642" alt="image" src="https://github.com/user-attachments/assets/cfa68e5c-a27b-410e-9adb-72676447d8b3" />

### Comparison of Normal vs Anomalous Buyer

| Metric                | Buyer 39 (Normal) | Buyer 564 (Anomalous) |
|----------------------|------------------|------------------------|
| **n_lots**           | 6                | 6                      |
| **n_suppliers**      | 6                | 4                      |
| **spend**            | 1,375,795.81     | 652,000.00             |
| **single_bid_rate**  | 0.1667           | 0.3333                 |
| **top_supplier_share** | 0.1667         | 0.3333                 |
| **cpv_hhi**          | 0.2778           | 1.0                    |
| **sb_norm**          | 0.1667           | 0.3333                 |
| **ts_norm**          | 0.1523           | 0.3282                 |
| **cpv_norm**         | 0.2157           | 1.0                    |
| **anomaly_score**    | 0.1722           | 0.4633                 |
| **label**            | normal           | anomalous              |

### What makes Buyer 564 anomalous?

1. **Supplier concentration is much higher**
   - Buyer 39: 6 lots → 6 suppliers (1 lot per supplier)
   - Buyer 564: 6 lots → only 4 suppliers  
   - *Pattern: repeat awards to the same suppliers → possible favoritism.*

2. **Top supplier accounts for double the share**
   - Normal: 16.7%  
   - Anomalous: 33.3%  
   - *A strong sign of buyer–supplier dependency.*

3. **Higher single-bid rate**
   - Normal: 16.7%  
   - Anomalous: 33.3%  
   - *Single bids reduce competition; consistently high share may signal manipulation.*

4. **Extreme CPV HHI (market concentration)**
   - Normal HHI: 0.28  
   - Anomalous HHI: 1.0 (max possible)  
   - *All purchases come from the same CPV sector → suspiciously narrow procurement scope.*

5. **Overall anomaly score is 2.7× higher**
   - Normal: 0.17  
   - Anomalous: 0.46  
   - *This pushes Buyer 564 across the anomaly threshold.*

### Pattern Mining Results
Using PrefixSpan on the cleaned and labeled subsets:

- **Normal buyers** exhibit stable recurring patterns. Examples:
  - `66|P0|NT_1|OPE`
  - `90|P5|NT_1|OPE`
  - `15|P0|NT_2-3|OPE`

  These suggest consistent procurement behavior across similar sectors and procedures.

- **Anomalous buyers** do not exhibit frequent repeating patterns at moderate support thresholds.  
  This is expected because anomalous procurement behavior tends to be irregular, diverse, and case-specific.

These preliminary results show that normal procurement exhibits consistent structural patterns, while anomalous procurement appears less structured, validating the anomaly scoring methodology.

## 4. Issues and Challenges Encountered

### Data Quality Issues
- High frequency of missing or zero award prices complicates statistical modeling.  
- Lack of bidder information for a significant fraction of lots.  
- Supplier information inconsistencies lead to fragmented supplier identities.

### Structural Challenges
- Extreme imbalance between high-degree and low-degree suppliers leads to noise in pattern discovery.  
- Many buyers have very small graphs that do not support meaningful structural pattern extraction.

### Pattern Mining Difficulties
- Anomalous patterns do not appear frequently across buyers, which makes traditional frequent-pattern mining less effective at default thresholds.  
- PrefixSpan requires carefully tuned support thresholds to reveal meaningful anomalous motifs.

### Computational Constraints
- Full subgraph mining (gSpan) is computationally intensive and impractical for the full dataset.  
- Only feasible on a small representative subset of graphs, requiring strategic sampling.


