# ==========================================
# EXPERIMENT GA - MBG SUKOLILO (ADVANCED)
# ==========================================
import pandas as pd
import random
import time
import osmnx as ox
import networkx as nx
import folium
import numpy as np
import requests
import json

# Set seed for reproducibility
random.seed(42)

# SETUP ENVIRONMENT
def setup_global_network():
    print("[*] Downloading global OSM road network...")
    place_name = "Sukolilo, Surabaya, Indonesia"
    G = ox.graph_from_place(place_name, network_type="drive")
    G = ox.truncate.largest_component(G, strongly=True)
    return G

def setup_cluster_data(sppg_name, df_sppg, df_schools_cluster, G_global):
    depot_row = df_sppg[df_sppg["nama"] == sppg_name].iloc[0]
    depot_data = {"nama": depot_row["nama"], "lat": depot_row["lat"], "lng": depot_row["lng"], "tipe": "DEPOT"}
    
    cluster_points = pd.concat([
        pd.DataFrame([depot_data]),
        df_schools_cluster.assign(tipe="SEKOLAH")
    ], ignore_index=True)

    def get_node(lat, lng):
        return ox.distance.nearest_nodes(G_global, lng, lat)
    
    cluster_points["node"] = cluster_points.apply(lambda row: get_node(row["lat"], row["lng"]), axis=1)
    nodes = cluster_points["node"].tolist()

    dist_matrix = {}
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j: dist_matrix[(i, j)] = 0
            else:
                try:
                    dist_matrix[(i, j)] = nx.shortest_path_length(G_global, nodes[i], nodes[j], weight="length")
                except nx.NetworkXNoPath:
                    dist_matrix[(i, j)] = 1e9
    return nodes, dist_matrix, cluster_points

# GA OPERATORS
def calculate_fitness(route, dist_matrix):
    total = dist_matrix[(0, route[0])] 
    for i in range(len(route) - 1):
        total += dist_matrix[(route[i], route[i + 1])]
    total += dist_matrix[(route[-1], 0)] 
    return total

# Selection Varian  
def selection_tournament(pop, dist_matrix, k=3):
    selected = random.sample(pop, k)
    return min(selected, key=lambda x: calculate_fitness(x, dist_matrix))

def selection_sus(pop, dist_matrix, n_select=1):
    fits = [1.0 / (calculate_fitness(ind, dist_matrix) + 1e-6) for ind in pop]
    total_fit = sum(fits)
    distance = total_fit / n_select
    start = random.uniform(0, distance)
    pointers = [start + i * distance for i in range(n_select)]
    selected = []
    for ptr in pointers:
        i = 0
        cum_fit = fits[0]
        while cum_fit < ptr and i < len(pop) - 1:
            i += 1
            cum_fit += fits[i]
        selected.append(pop[i])
    return selected[0]

def selection_roulette(pop, dist_matrix):
    fits = [1.0 / (calculate_fitness(ind, dist_matrix) + 1e-6) for ind in pop]
    total = sum(fits)
    probs = [f/total for f in fits]
    return random.choices(pop, weights=probs, k=1)[0]

# Crossover Varian  
def crossover_ox1(p1, p2):
    size = len(p1)
    if size < 2: return p1[:]
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = p1[a:b]
    p2_idx = 0
    for i in range(size):
        if child[i] is None:
            while p2[p2_idx] in child:
                p2_idx += 1
            child[i] = p2[p2_idx]
    return child

def crossover_pmx(p1, p2):
    size = len(p1)
    if size < 2: return p1[:]
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = p1[a:b]
    mapping = {p1[i]: p2[i] for i in range(a, b)}
    for i in range(size):
        if child[i] is None:
            val = p2[i]
            while val in child[a:b]:
                val = mapping[val]
            child[i] = val
    return child

# Mutation Varian  
def mutation_inversion(route, rate=0.2):
    new_route = route[:]
    if len(new_route) < 2: return new_route
    if random.random() < rate:
        a, b = sorted(random.sample(range(len(new_route)), 2))
        new_route[a:b] = reversed(new_route[a:b])
    return new_route

def mutation_swap(route, rate=0.2):
    new_route = route[:]
    if len(new_route) < 2: return new_route
    if random.random() < rate:
        i, j = random.sample(range(len(new_route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

# CORE GA ENGINE
def run_ga(nodes, dist_matrix, sel_op, cross_op, mut_op, pop_size=50, generations=100, mut_rate=0.2):
    num_nodes = len(nodes)
    if num_nodes <= 1: return [0, 0], 0
    base = list(range(1, num_nodes))
    pop = [random.sample(base, len(base)) for _ in range(pop_size)]
    best = min(pop, key=lambda x: calculate_fitness(x, dist_matrix))
    for _ in range(generations):
        new_pop = [best]
        while len(new_pop) < pop_size:
            p1, p2 = sel_op(pop, dist_matrix), sel_op(pop, dist_matrix)
            child = cross_op(p1, p2)
            child = mut_op(child, mut_rate)
            new_pop.append(child)
        pop = new_pop
        curr_best = min(pop, key=lambda x: calculate_fitness(x, dist_matrix))
        if calculate_fitness(curr_best, dist_matrix) < calculate_fitness(best, dist_matrix):
            best = curr_best
    return [0] + best + [0], calculate_fitness(best, dist_matrix)

# 4. OSRM GEOMETRY HELPER
def get_osrm_geometry(coords):
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
    url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("code") == "Ok":
            return [(lat, lon) for lon, lat in data["routes"][0]["geometry"]["coordinates"]]
    except: pass
    return coords # fallback

# MAIN 
if __name__ == "__main__":
    df_schools = pd.read_csv("sekolah_terklaster.csv")
    df_sppg = pd.read_csv("sppg_sukolilo.csv")
    G_global = setup_global_network()
    unique_sppgs = sorted(df_schools["sppg_terdekat"].unique())

    cluster_data = {s: setup_cluster_data(s, df_sppg, df_schools[df_schools["sppg_terdekat"] == s], G_global) for s in unique_sppgs}

    scenarios = [
        {"name": "Tournament + OX1 + Inversion", "sel": selection_tournament, "cross": crossover_ox1, "mut": mutation_inversion},
        {"name": "SUS + PMX + Swap", "sel": selection_sus, "cross": crossover_pmx, "mut": mutation_swap},
        {"name": "Roulette + OX1 + Inversion", "sel": selection_roulette, "cross": crossover_ox1, "mut": mutation_inversion},
        {"name": "SUS + OX1 + Inversion", "sel": selection_sus, "cross": crossover_ox1, "mut": mutation_inversion},
    ]

    all_performance_data = []

    print("\n" + "="*60)
    print("RUNNING GA SCENARIO COMPARISON (DETAILED)")
    print("="*60)

    for scen in scenarios:
        print(f"\n[>] Testing: {scen['name']}")
        start_time = time.time()
        for sppg_name in unique_sppgs:
            nodes, dist_matrix, _ = cluster_data[sppg_name]
            _, dist = run_ga(nodes, dist_matrix, scen['sel'], scen['cross'], scen['mut'])
            all_performance_data.append({
                "Scenario": scen['name'],
                "Cluster": sppg_name[-25:],
                "Distance (km)": round(dist/1000, 2)
            })
        print(f"    Done in {time.time() - start_time:.2f}s")

    # Display Pivot Table
    df_perf = pd.DataFrame(all_performance_data)
    pivot_perf = df_perf.pivot(index="Cluster", columns="Scenario", values="Distance (km)")
    print("\n" + "="*60)
    print("PERFORMANCE PER CLUSTER (DISTANCE IN KM)")
    print("="*60)
    print(pivot_perf)

    # Summary Table
    summary = df_perf.groupby("Scenario")["Distance (km)"].sum().reset_index()
    print("\n" + "="*60)
    print("TOTAL DISTANCE SUMMARY")
    print("="*60)
    print(summary)

    # Visualization of the BEST overall scenario
    best_scen_name = summary.loc[summary["Distance (km)"].idxmin(), "Scenario"]
    best_scen = next(s for s in scenarios if s["name"] == best_scen_name)
    
    print(f"\n[*] Generating final map for best scenario: {best_scen_name}")
    m = folium.Map(location=[-7.2991, 112.7838], zoom_start=13, tiles='CartoDB Voyager')
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue']

    for idx, sppg_name in enumerate(unique_sppgs):
        nodes, dist_matrix, points = cluster_data[sppg_name]
        best_route, d = run_ga(nodes, dist_matrix, best_scen['sel'], best_scen['cross'], best_scen['mut'], generations=150)
        
        c = colors[idx % len(colors)]
        group = folium.FeatureGroup(name=f"{sppg_name[-25:]} ({round(d/1000, 2)} km)")
        
        # Get actual lat/lng sequence for OSRM
        route_coords = [(points.iloc[i]["lat"], points.iloc[i]["lng"]) for i in best_route]
        asphalt_path = get_osrm_geometry(route_coords)
        
        folium.PolyLine(asphalt_path, color=c, weight=5, opacity=0.8, tooltip=f"Rute {sppg_name}").add_to(group)

        for i, pt_idx in enumerate(best_route[:-1]):
            row = points.iloc[pt_idx]
            label = "DEPOT" if row["tipe"] == "DEPOT" else f"#{i} {row['nama_sekolah']}"
            icon_color = 'black' if row["tipe"] == "DEPOT" else c
            icon_type = 'industry' if row["tipe"] == "DEPOT" else 'graduation-cap'
            
            folium.Marker(
                [row["lat"], row["lng"]],
                popup=label,
                tooltip=label,
                icon=folium.Icon(color=icon_color, icon=icon_type, prefix='fa')
            ).add_to(group)
        group.add_to(m)

    folium.LayerControl().add_to(m)
    m.save("final_comparison_ga_map.html")
    print(f"\n[*] Final map saved: final_comparison_ga_map.html")
