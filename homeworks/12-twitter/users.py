# Reloading necessary libraries
import pandas as pd
import networkx as nx


users_df = pd.read_csv('data/users.csv')

# Retain only the specified columns
users_df = users_df[["id", "location", "name", "followers_count", "statuses_count", 
                     "time_zone", "verified", "lang", "screen_name", "created_at", 
                     "favourites_count", "friends_count", "listed_count"]]

# Fill NaNs: strings with "nan" and numerics with 0
users_df.fillna({
    "location": "nan",
    "name": "nan",
    "time_zone": "nan",
    "lang": "nan",
    "screen_name": "nan",
    "created_at": "nan",
    "verified": False  # Verified is boolean, treated separately
}, inplace=True)
numeric_columns = ["followers_count", "statuses_count", "favourites_count", 
                   "friends_count", "listed_count"]
users_df[numeric_columns] = users_df[numeric_columns].fillna(0)

# Create a graph with users as nodes
G = nx.DiGraph()

# Add nodes for users with attributes
for _, row in users_df.iterrows():
    G.add_node(
        row["id"],
        location=row["location"],
        name=row["name"],
        followers_count=row["followers_count"],
        statuses_count=row["statuses_count"],
        time_zone=row["time_zone"],
        verified=row["verified"],
        lang=row["lang"],
        screen_name=row["screen_name"],
        created_at=row["created_at"],
        favourites_count=row["favourites_count"],
        friends_count=row["friends_count"],
        listed_count=row["listed_count"]
    )

# Calculate PageRank and add it as a node attribute
pagerank = nx.pagerank(G)
nx.set_node_attributes(G, pagerank, "pagerank")

# Perform HITS clustering (authority and hub scores)
hits = nx.hits(G)
nx.set_node_attributes(G, hits[0], "authority_score")
nx.set_node_attributes(G, hits[1], "hub_score")

# Save the graph to a GraphML file for Gephi
output_path = 'users_graph.graphml'
nx.write_graphml(G, output_path)

