import pandas as pd
import osmnx as ox
import networkx as nx
import json
import os

# Load data
df_schools = pd.read_csv("sekolah_terklaster.csv")
df_sppg = pd.read_csv("sppg_sukolilo.csv")

# Download graph
place_name = "Sukolilo, Surabaya, Indonesia"
G = ox.graph_from_place(place_name, network_type="drive")
G = ox.truncate.largest_component(G, strongly=True)

# Pick one cluster
sppg_name = "SPPG Kota Surabaya Sukolilo Semolowaru"
df_cluster = df_schools[df_schools["sppg_terdekat"] == sppg_name]

# Setup points
depot_row = df_sppg[df_sppg["nama"] == sppg_name].iloc[0]
depot_data = {"nama": depot_row["nama"], "lat": depot_row["lat"], "lng": depot_row["lng"]}
cluster_points = pd.concat([pd.DataFrame([depot_data]), df_cluster], ignore_index=True)

# Map nodes
def get_node(lat, lng):
    return ox.distance.nearest_nodes(G, lng, lat)

cluster_points["node"] = cluster_points.apply(lambda row: get_node(row["lat"], row["lng"]), axis=1)
nodes = cluster_points["node"].tolist()

# GA matrix calculation
print(f"--- Comparison for {sppg_name} ---")
print(f"{'From':<20} | {'To':<20} | {'GA (OSMnx)':<12} | {'JSON (OSRM)':<12}")
print("-" * 75)

with open("matriks_jarak/matrix_SPPG_Kota_Surabaya_Sukolilo_Semolowaru.json", "r") as f:
    json_data = json.load(f)
    json_matrix = json_data["distance_matrix_meters"]

for i in range(len(nodes)):
    for j in range(len(nodes)):
        if i != j:
            try:
                ga_dist = nx.shortest_path_length(G, nodes[i], nodes[j], weight="length")
            except:
                ga_dist = -1
            
            json_dist = json_matrix[i][j]
            
            if i == 0 or j == 0 or i < 3: # Just show some samples
                name_i = str(cluster_points.iloc[i].get("nama_sekolah", cluster_points.iloc[i]["nama"]))[:18]
                name_j = str(cluster_points.iloc[j].get("nama_sekolah", cluster_points.iloc[j]["nama"]))[:18]
                print(f"{name_i:<20} | {name_j:<20} | {ga_dist:>10.1f}m | {json_dist:>10.1f}m")
