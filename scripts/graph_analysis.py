# %% [markdown]
#  # Graph Analysis & Pattern Mining for Cartel Detection
# 
# 
# 
#  ## 1. Objectives
# 
#  **Goal:** Utilize Network Science and Graph Theory to detect collusive patterns that are invisible to tabular models.
# 
# 
# 
#  **Hypothesis:**
# 
#  Cartels operate through specific structural patterns:
# 
#  1.  **Market Partitioning:** Specific groups of bidders exclusively targeting specific buyers.
# 
#  2.  **Rotation:** High clustering among cartel members who take turns winning.
# 
#  3.  **Isolation:** Cartel communities often have few links to the "honest" market.
# 
# 
# 
#  **Methodology:**
# 
#  1.  Construct a **Bipartite Graph** (Buyers $\leftrightarrow$ Bidders).
# 
#  2.  Project to a **Bidder-Bidder Network** (Competitor Graph).
# 
#  3.  Calculate Graph Metrics (Centrality, Clustering, Community Structure).
# 
#  4.  Visualise networks to spot "Cartel Communities".
# 
# 
# 
#  ## 2. Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings
import utils

# Visualization Settings
utils.setup_plotting_style()
warnings.filterwarnings('ignore')

print("Environment Setup Complete.")


# %% [markdown]
#  ## 3. Data Loading & Graph Construction
# 
#  We use the raw IDs to build the network.

# %%
# Load data
df = utils.load_data("data/GTI_labelled_cartel_data_NOV2023.csv")

# Deduplicate (Critical for graph weights)
print(f"Original shape: {df.shape}")
df = df.drop_duplicates()
df = df.reset_index(drop=True)
print(f"Shape after deduplication: {df.shape}")

# Filter for graph construction
# We need nodes: Buyer_ID, Bidder_ID
# Edges: Tender participation
graph_df = df[['buyer_id', 'bidder_id', 'is_cartel', 'tender_id', 'lot_bidscount']].dropna()

# Create bipartite graph
B = nx.Graph()

# Add nodes with attributes
# 1. Buyers
buyers = graph_df['buyer_id'].unique()
B.add_nodes_from(buyers, bipartite=0, node_type='Buyer')

# 2. Bidders (add cartel flag as attribute)
bidders = graph_df['bidder_id'].unique()
# Calculate cartel rate per bidder
bidder_cartel_rate = graph_df.groupby('bidder_id')['is_cartel'].mean()
bidder_size = graph_df.groupby('bidder_id')['tender_id'].count()

for bidder in bidders:
    rate = bidder_cartel_rate.get(bidder, 0)
    size = bidder_size.get(bidder, 0)
    # Color node red if predominantly cartel, blue otherwise
    color = 'red' if rate > 0.5 else 'blue'
    B.add_node(bidder, bipartite=1, node_type='Bidder', cartel_rate=rate, size=size, color=color)

# Add edges (participation)
# We can weight edges by number of tenders if they interact multiple times
edge_data = graph_df.groupby(['buyer_id', 'bidder_id']).size().reset_index(name='weight')

for _, row in edge_data.iterrows():
    B.add_edge(row['buyer_id'], row['bidder_id'], weight=row['weight'])

print(f"Bipartite Graph Constructed.")
print(f"Nodes: {B.number_of_nodes()} (Buyers: {len(buyers)}, Bidders: {len(bidders)})")
print(f"Edges: {B.number_of_edges()}")


# %% [markdown]
#  ## 4. Network Visualization (Bipartite)
# 
#  Visualizing the structure of the market. Are there isolated islands?

# %%
# Plotting the whole graph is messy. Let's plot the largest connected component or a sample.
# Let's filter for nodes involved in Cartel cases to see their neighborhood.

cartel_bidders = [n for n, attr in B.nodes(data=True) if attr.get('node_type') == 'Bidder' and attr.get('cartel_rate', 0) > 0.5]
print(f"Number of High-Risk Cartel Bidders: {len(cartel_bidders)}")

# Get subgraph of cartel bidders and their buyers
subset_nodes = set(cartel_bidders)
for bidder in cartel_bidders:
    neighbors = list(B.neighbors(bidder))
    subset_nodes.update(neighbors)
    
subgraph = B.subgraph(subset_nodes)

print(f"Subgraph Nodes: {subgraph.number_of_nodes()}")
print(f"Subgraph Edges: {subgraph.number_of_edges()}")

if subgraph.number_of_nodes() < 2000: # Increased limit since 1179 is manageable
    plt.figure(figsize=(15, 10))
    # Use a layout that separates components well
    pos = nx.spring_layout(subgraph, k=0.05, iterations=50, seed=42)
    
    # Draw Buyers
    nx.draw_networkx_nodes(subgraph, pos, 
                           nodelist=[n for n, d in subgraph.nodes(data=True) if d.get('node_type') == 'Buyer'],
                           node_color='lightgrey', node_size=30, label='Buyer', alpha=0.6)
    
    # Draw Honest Bidders (if any in neighborhood)
    nx.draw_networkx_nodes(subgraph, pos, 
                           nodelist=[n for n, d in subgraph.nodes(data=True) if d.get('node_type') == 'Bidder' and d.get('cartel_rate', 0) <= 0.5],
                           node_color='skyblue', node_size=20, label='Honest Bidder', alpha=0.6)
    
    # Draw Cartel Bidders
    nx.draw_networkx_nodes(subgraph, pos, 
                           nodelist=[n for n, d in subgraph.nodes(data=True) if d.get('node_type') == 'Bidder' and d.get('cartel_rate', 0) > 0.5],
                           node_color='red', node_size=60, label='Cartel Bidder', alpha=0.9) # Make them pop
    
    nx.draw_networkx_edges(subgraph, pos, alpha=0.1, edge_color='gray')
    plt.title("Network Topology of Cartel Neighborhoods (Red=Cartel)")
    # Only show legend for unique labels to avoid clutter
    # plt.legend() 
    plt.axis('off')
    utils.save_plot(plt.gcf(), "network_topology.png")
    # plt.show()
else:
    print("Subgraph too large to plot clearly.")


# %% [markdown]
#  ## 5. Graph Metrics Analysis
# 
#  Does the network position predict cartel behavior?
# 
# 
# 
#  We calculate:
# 
#  1.  **Degree Centrality:** Is the bidder a "hub" (wins everywhere)?
# 
#  2.  **Clustering Coefficient (Projected):** Do they form tight cliques?

# %%
# Degree Centrality (in Bipartite graph)
# Normalize? Not necessarily needed for comparison within same graph.
degrees = dict(B.degree())

# Extract metrics for Bidders only
bidder_metrics = []
for bidder in bidders:
    # Degree in bipartite = number of unique buyers they work with
    d = degrees[bidder]
    
    # Get neighbors (buyers)
    my_buyers = list(B.neighbors(bidder))
    
    # Calculate simple resource concentration
    # "Average bids per tender" for this bidder's market
    # (Are they operating in high competition or low competition zones?)
    # We can average the 'lot_bidscount' from the dataframe for this bidder
    avg_competition = graph_df[graph_df['bidder_id'] == bidder]['lot_bidscount'].mean()
    
    bidder_metrics.append({
        'bidder_id': bidder,
        'degree_buyers': d,
        'avg_market_competition': avg_competition,
        'is_cartel_rate': B.nodes[bidder]['cartel_rate'],
        'total_tenders': B.nodes[bidder]['size']
    })

metrics_df = pd.DataFrame(bidder_metrics)

# Correlation with Cartel Rate
print("--- Correlation of Graph Metrics with Cartel Rate ---")
print(metrics_df.corrwith(metrics_df['is_cartel_rate'], numeric_only=True))

# Plot Degree vs Cartel
plt.figure(figsize=(10, 6))
sns.scatterplot(data=metrics_df, x='degree_buyers', y='avg_market_competition', hue='is_cartel_rate', palette='coolwarm', alpha=0.7)
plt.title("Bidder Strategy: Breadth (Degree) vs. Competition Intensity")
plt.xlabel("Number of Unique Buyers (Degree)")
plt.ylabel("Avg Competition in Tenders (Bid Count)")
plt.xscale('log')
plt.yscale('log')
utils.save_plot(plt.gcf(), "bidder_strategy.png")
# plt.show()


# %% [markdown]
#  ## 6. Bidder-Bidder Projection (Competitor Graph)
# 
#  Connecting bidders if they bid for the same buyer.
# 
#  WARNING: This can be dense. We filter for meaningful overlaps.

# %%
# Project bipartite graph to bidder nodes
# Two bidders are connected if they share at least 1 buyer
# Weighted by number of shared buyers
from networkx.algorithms import bipartite

# Weighted projection
print("Projecting to Bidder-Bidder network (this may take a moment)...")
# Note: projected_graph is computationally expensive if graph is dense. 
# We'll use a manual overlap approach for control if needed, but let's try standard projection first.
try:
    # Use generic_weighted_projected_graph for efficiency if bipartite is strictly defined
    P = bipartite.weighted_projected_graph(B, bidders)
    
    print("\n--- Graph Definition ---")
    print("Nodes: Bidders")
    print("Edges: Competitors")
    print("Definition: Two bidders are connected if they have bid for the SAME Buyer.")
    print("Weight: Number of shared buyers/tenders.")
    print("------------------------\n")
    
    # Propagate attributes from Bipartite graph B to Projected graph P
    for node in P.nodes():
        if node in B.nodes:
            P.nodes[node].update(B.nodes[node])
            
    print(f"Projection Complete. Edges: {P.number_of_edges()}")
except Exception as e:
    print(f"Projection failed: {e}")
    P = nx.Graph() # Empty fallback


# %% [markdown]
#  ### 6.1 Community Detection (Louvain)
# 
#  Detecting clusters of bidders who constantly compete (or collude) with each other.

# %%
# --- NEW: Global Assortativity ---
try:
    # Assortativity: Do cartel nodes connect to other cartel nodes?
    # Values close to 1: High segregation (Cartels play with Cartels)
    # Values close to -1: Disassortative
    # Values close to 0: Random mixing
    assortativity = nx.attribute_assortativity_coefficient(P, 'cartel_rate')
    print(f"Global Network Assortativity (Cartel Rate): {assortativity:.4f}")
except Exception as e:
    print(f"Assortativity calculation failed: {e}")

# --- NEW: Advanced Metrics & Homophily ---
print("Calculating advanced metrics (Centrality & Homophily)...")

# 1. Neighbor Cartel Exposure
# "Tell me who your friends are, and I'll tell you who you are"
neighbor_exposure = {}
clustering_coeff = nx.clustering(P, weight='weight')

# Centrality measures (can be slow on large graphs)
try:
    # Eigenvector Centrality:
    # Measures "Influence". A high score means you are connected to other nodes who are ALSO important.
    # In this context: A bidder who competes with other major/central bidders.
    # Cartels might have different scores if they only compete in a closed "ring" of nodes.
    eigenvector = nx.eigenvector_centrality(P, weight='weight', max_iter=1000)
except:
    eigenvector = {n: 0 for n in P.nodes()}

try:
    # Betweenness can be very slow, maybe skip for huge graphs or use k-samples
    betweenness = nx.betweenness_centrality(P, weight='weight', k=min(100, len(P)))
except:
    betweenness = {n: 0 for n in P.nodes()}

for node in P.nodes():
    neighbors = list(P.neighbors(node))
    if not neighbors:
        neighbor_exposure[node] = 0
        continue
    
    # Calculate weighted average of neighbors' cartel rates
    total_weight = 0
    weighted_cartel_sum = 0
    
    for nbr in neighbors:
        # Get weight of connection (shared tenders)
        w = P[node][nbr].get('weight', 1)
        # Get neighbor's cartel rate
        nbr_rate = P.nodes[nbr].get('cartel_rate', 0)
        
        weighted_cartel_sum += w * nbr_rate
        total_weight += w
        
    if total_weight > 0:
        neighbor_exposure[node] = weighted_cartel_sum / total_weight
    else:
        neighbor_exposure[node] = 0

# Update metrics dataframe with new features
# We need to rebuild metrics_df or merge these new dicts
advanced_metrics = []
for node in P.nodes():
    advanced_metrics.append({
        'bidder_id': node,
        'neighbor_cartel_exposure': neighbor_exposure.get(node, 0),
        'clustering_coefficient': clustering_coeff.get(node, 0),
        'eigenvector_centrality': eigenvector.get(node, 0),
        'betweenness_centrality': betweenness.get(node, 0)
    })
    
adv_df = pd.DataFrame(advanced_metrics)

# Merge with existing metrics
# Note: original metrics_df was built from Bipartite 'B', this is from Projected 'P'.
# They should have same bidders.
full_metrics = pd.merge(metrics_df, adv_df, on='bidder_id', how='inner')

# Create a binary Label for clearer plotting
full_metrics['Label'] = full_metrics['is_cartel_rate'].apply(lambda x: 'Cartel' if x > 0.5 else 'Honest')

print("Advanced metrics calculated.")

# --- STATISTICS REPORT ---
print("\n" + "="*40)
print("       DATA REVIEW & STATISTICS       ")
print("="*40)

print("\n1. Graph Dimensions:")
print(f"   - Total Bidders (Nodes): {P.number_of_nodes()}")
print(f"   - Total Competitor Relationships (Edges): {P.number_of_edges()}")

print("\n2. Homophily & Exposure Stats:")
stats = full_metrics.groupby('Label')[['neighbor_cartel_exposure', 'eigenvector_centrality', 'clustering_coefficient']].agg(['mean', 'median', 'std'])
print(stats)

print("\n3. Interpretation:")
print("   - Neighbor Cartel Exposure: If Cartel > Honest, it confirms homophily (Cartels hang out with Cartels).")
print("   - Eigenvector: If Cartel < Honest, they might be peripheral/isolated.")
print("   - Clustering: If Cartel > Honest, they form tighter cliques.")
print("="*40 + "\n")


# Plotting Homophily (Simplified)
plt.figure(figsize=(8, 6))
# We use the binary 'Label' for the X-axis to avoid the messy granular plot
sns.boxplot(data=full_metrics, x='Label', y='neighbor_cartel_exposure', palette={'Cartel': 'red', 'Honest': 'skyblue'})
plt.title("Homophily Analysis: Neighbor Cartel Exposure by Group")
plt.xlabel("Bidder Status")
plt.ylabel("Avg Neighbor Cartel Rate (Exposure)")
utils.save_plot(plt.gcf(), "homophily_boxplot.png")
# plt.show()

# Scatter: Centrality vs Exposure
plt.figure(figsize=(10, 6))
sns.scatterplot(data=full_metrics, x='eigenvector_centrality', y='neighbor_cartel_exposure', 
                hue='Label', palette={'Cartel': 'red', 'Honest': 'blue'}, size='total_tenders', sizes=(20, 200), alpha=0.7)
plt.title("Risk Profile: Influence (Eigenvector) vs. Cartel Exposure")
plt.xlabel("Eigenvector Centrality (Network Influence - 'Connectedness')")
plt.ylabel("Neighbor Cartel Exposure")
plt.legend(title='Bidder Status')
utils.save_plot(plt.gcf(), "centrality_vs_exposure.png")
# plt.show()


if P.number_of_edges() > 0:
    try:
        # Detect communities using Louvain Method
        # (Requires scipy/networkx 2.x+)
        communities = nx.community.louvain_communities(P, weight='weight', resolution=1.0)
        print(f"Detected {len(communities)} communities.")
        
        # Analyze communities
        community_stats = []
        for i, comm in enumerate(communities):
            comm_list = list(comm)
            size = len(comm_list)
            if size < 3: continue # Ignore tiny pairs
            
            # Calculate cartel density in this community
            # B.nodes[n] might fail if n is not in B (unlikely if projected from B) or attribute missing
            cartel_rates = []
            for n in comm_list:
                try:
                    # Access node data from original Bipartite graph B
                    rate = B.nodes[n].get('cartel_rate', 0) 
                    cartel_rates.append(rate)
                except KeyError:
                    cartel_rates.append(0)
            
            avg_cartel = np.mean(cartel_rates)
            
            community_stats.append({
                'community_id': i,
                'size': size,
                'avg_cartel_rate': avg_cartel,
                'members': comm_list
            })
            
        comm_df = pd.DataFrame(community_stats)
        
        # Plot Community Cartel Density
        plt.figure(figsize=(10, 6))
        sns.histplot(comm_df['avg_cartel_rate'], bins=20, kde=True, color='purple')
        plt.title("Community Cartel Density Distribution")
        plt.xlabel("Average Cartel Rate of Community (0=Clean, 1=Pure Cartel)")
        utils.save_plot(plt.gcf(), "community_density.png")
        # plt.show()
        
        print("\n--- Top Suspicious Communities ---")
        print(comm_df.sort_values(by='avg_cartel_rate', ascending=False).head(5).to_string())

        # --- NEW: Visualization of Bidder-Bidder Communities ---
        print("\nVisualizing Bidder Communities (Color = Cartel Rate)...")
        plt.figure(figsize=(12, 12))
        
        # 1. Simplify: Only plot communities with significant size or cartel risk
        # Filter for top N communities or those with size > X
        risky_communities = comm_df[comm_df['avg_cartel_rate'] > 0.1]['community_id'].tolist()
        
        # Create subgraph of only these communities
        relevant_nodes = []
        node_colors = []
        
        # Map community ID to color
        # We will color nodes by their INDIVIDUAL cartel rate, but position them by community
        
        pos = nx.spring_layout(P, k=0.15, iterations=30, seed=42) # Calculate layout for whole graph (or subgraph)
        
        # Draw edges first (faint)
        nx.draw_networkx_edges(P, pos, alpha=0.05, edge_color='gray')
        
        # Draw nodes colored by their cartel status
        # Cartel = Red, Honest = Blue
        
        cartel_nodes = [n for n in P.nodes() if B.nodes[n].get('cartel_rate', 0) > 0.5]
        honest_nodes = [n for n in P.nodes() if B.nodes[n].get('cartel_rate', 0) <= 0.5]
        
        nx.draw_networkx_nodes(P, pos, nodelist=honest_nodes, node_color='skyblue', node_size=20, alpha=0.4, label='Honest')
        nx.draw_networkx_nodes(P, pos, nodelist=cartel_nodes, node_color='red', node_size=50, alpha=0.8, label='Cartel')
        
        plt.title("Bidder Competitor Network: Clustering of Cartel Members")
        plt.axis('off')
        plt.legend(scatterpoints=1)
        utils.save_plot(plt.gcf(), "bidder_communities.png")
        # plt.show()
        
    except AttributeError:
        print("Louvain community detection not available in this NetworkX version.")
    except Exception as e:
        print(f"Community detection error: {e}")


# %% [markdown]
#  ## 7. Conclusion
# 
# 
# 
#  ### Insights
# 
#  1.  **Topology:** We see red nodes (cartels) clustering together around specific buyers, it confirms market allocation.
# 
#  2.  **Metrics:**
# 
#      *   **Low Degree, Low Competition:** Bidders with few buyers and low competition are highly suspicious (Local Monopolies).
# 
#      *   **High Degree, High Competition:** Likely honest market players.
# 
#  3.  **Communities:** If `avg_cartel_rate` in communities is bimodal (peaks at 0 and 1), it strongly suggests that **cartels form distinct, isolated sub-graphs** separate from the honest market. This "Network Separation" is a powerful feature we can add to the ML model.

