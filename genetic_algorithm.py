import pandas as pd
import random
import osmnx as ox
import networkx as nx
import folium
import requests
from copy import deepcopy

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
    cluster_points = pd.concat([pd.DataFrame([depot_data]), df_schools_cluster.assign(tipe="SEKOLAH")], ignore_index=True)
    
    def get_node(lat, lng): return ox.distance.nearest_nodes(G_global, lng, lat)
    cluster_points["node"] = cluster_points.apply(lambda row: get_node(row["lat"], row["lng"]), axis=1)
    nodes = cluster_points["node"].tolist()

    dist_matrix = {}
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j: dist_matrix[(i, j)] = 0
            else:
                try: dist_matrix[(i, j)] = nx.shortest_path_length(G_global, nodes[i], nodes[j], weight="length")
                except nx.NetworkXNoPath: dist_matrix[(i, j)] = 1e9
    return nodes, dist_matrix, cluster_points

# GA OPERATORS
def calculate_fitness(route, dist_matrix):
    total = dist_matrix[(0, route[0])] 
    for i in range(len(route) - 1): total += dist_matrix[(route[i], route[i + 1])]
    total += dist_matrix[(route[-1], 0)] 
    return total

def selection_tournament(pop, dist_matrix, k=2):
    selected = random.sample(pop, k)
    return min(selected, key=lambda x: calculate_fitness(x, dist_matrix))

def selection_sus(pop, dist_matrix, n_select=1):
    fits = [1.0 / (calculate_fitness(ind, dist_matrix) + 1e-6) for ind in pop]
    total_fit = sum(fits)
    dist = total_fit / n_select
    ptr = random.uniform(0, dist)
    i, cum_fit = 0, fits[0]
    while cum_fit < ptr:
        i += 1
        cum_fit += fits[i]
    return pop[i]

def selection_roulette(pop, dist_matrix):
    fits = [1.0 / (calculate_fitness(ind, dist_matrix) + 1e-6) for ind in pop]
    return random.choices(pop, weights=fits, k=1)[0]

def crossover_ox1(p1, p2):
    size = len(p1)
    if size < 2: return p1[:]
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = p1[a:b]
    p2_idx = 0
    for i in range(size):
        if child[i] is None:
            while p2[p2_idx] in child: p2_idx += 1
            child[i] = p2[p2_idx]
    return child

def mutation_inversion(route, rate=0.2):
    if len(route) < 2 or random.random() > rate: return route[:]
    a, b = sorted(random.sample(range(len(route)), 2))
    res = route[:]
    res[a:b] = reversed(res[a:b])
    return res

def mutation_swap(route, rate=0.2):
    if len(route) < 2 or random.random() > rate: return route[:]
    i, j = random.sample(range(len(route)), 2)
    res = route[:]
    res[i], res[j] = res[j], res[i]
    return res

def mutation_scramble(route, rate=0.2):
    if len(route) < 2 or random.random() > rate: return route[:]
    a, b = sorted(random.sample(range(len(route)), 2))
    res = route[:]
    subset = res[a:b]
    random.shuffle(subset)
    res[a:b] = subset
    return res

# CORE GA ENGINE
def run_ga(nodes, dist_matrix, sel_op, cross_op, mut_op, pop_size=10, generations=15, mut_rate=0.2, init_pop=None):
    num_nodes = len(nodes)
    if num_nodes <= 1: return [0, 0], 0
    
    # Use provided initial population or create new one
    pop = init_pop if init_pop else [random.sample(list(range(1, num_nodes)), num_nodes-1) for _ in range(pop_size)]
    best = min(pop, key=lambda x: calculate_fitness(x, dist_matrix))

    for _ in range(generations):
        new_pop = [best] # Elitism
        while len(new_pop) < pop_size:
            p1, p2 = sel_op(pop, dist_matrix), sel_op(pop, dist_matrix)
            child = mut_op(cross_op(p1, p2), mut_rate)
            new_pop.append(child)
        pop = new_pop
        curr_best = min(pop, key=lambda x: calculate_fitness(x, dist_matrix))
        if calculate_fitness(curr_best, dist_matrix) < calculate_fitness(best, dist_matrix):
            best = curr_best
    return [0] + best + [0], calculate_fitness(best, dist_matrix)

# OSRM & EXECUTION
def get_osrm_geometry(coords):
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
    try:
        r = requests.get(f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson", timeout=5)
        return [(lat, lon) for lon, lat in r.json()["routes"][0]["geometry"]["coordinates"]]
    except: return coords

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    df_schools, df_sppg = pd.read_csv("sekolah_terklaster.csv"), pd.read_csv("sppg_sukolilo.csv")
    G_global = setup_global_network()
    unique_sppgs = sorted(df_schools["sppg_terdekat"].unique())
    cluster_data = {s: setup_cluster_data(s, df_sppg, df_schools[df_schools["sppg_terdekat"] == s], G_global) for s in unique_sppgs}

    scenarios = [
        {"name": "Tourn + OX1 + Inv", "sel": selection_tournament, "cross": crossover_ox1, "mut": mutation_inversion},
        {"name": "Roul + OX1 + Inv", "sel": selection_roulette, "cross": crossover_ox1, "mut": mutation_inversion},
        {"name": "SUS + OX1 + Inv", "sel": selection_sus, "cross": crossover_ox1, "mut": mutation_inversion},
        {"name": "Tourn + OX1 + Swap", "sel": selection_tournament, "cross": crossover_ox1, "mut": mutation_swap},
        {"name": "Roul + OX1 + Swap", "sel": selection_roulette, "cross": crossover_ox1, "mut": mutation_swap},
        {"name": "SUS + OX1 + Swap", "sel": selection_sus, "cross": crossover_ox1, "mut": mutation_swap},
        {"name": "Tourn + OX1 + Scramble", "sel": selection_tournament, "cross": crossover_ox1, "mut": mutation_scramble},
        {"name": "Roul + OX1 + Scramble", "sel": selection_roulette, "cross": crossover_ox1, "mut": mutation_scramble},
        {"name": "SUS + OX1 + Scramble", "sel": selection_sus, "cross": crossover_ox1, "mut": mutation_scramble},
    ]

    all_performance = []
    
    # Pre-generate initial populations for each cluster to ensure fairness
    initial_populations = {}
    random.seed(42)
    for s in unique_sppgs:
        n_nodes = len(cluster_data[s][0])
        base_nodes = list(range(1, n_nodes))
        initial_populations[s] = [random.sample(base_nodes, len(base_nodes)) for _ in range(15)]

    print("\n" + "="*70)
    print("GA PERFORMANCE COMPARISON (CONSTRAINED BUDGET: 15 Pop, 20 Gen)")
    print("="*70)

    for scen in scenarios:
        for sppg_name in unique_sppgs:
            nodes, dist_matrix, _ = cluster_data[sppg_name]
            # Use lower budget to reveal which operator combination is more efficient
            _, dist = run_ga(nodes, dist_matrix, scen['sel'], scen['cross'], scen['mut'], 
                             pop_size=15, generations=20, init_pop=deepcopy(initial_populations[sppg_name]))
            all_performance.append({"Scenario": scen['name'], "Cluster": sppg_name[-20:], "Dist (km)": round(dist/1000, 2)})

    df_perf = pd.DataFrame(all_performance)
    print(df_perf.pivot(index="Cluster", columns="Scenario", values="Dist (km)"))

    summary = df_perf.groupby("Scenario")["Dist (km)"].sum().reset_index().sort_values("Dist (km)")
    print("\n" + "="*70)
    print("RANKING SKENARIO (TOTAL JARAK)")
    print("="*70)
    print(summary)

    best_scen = next(s for s in scenarios if s["name"] == summary.iloc[0]["Scenario"])
    print(f"\n[*] Visualizing Best Scen: {best_scen['name']}")
    m = folium.Map(location=[-7.2991, 112.7838], zoom_start=13, tiles='CartoDB Voyager')
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue']

    for idx, sppg_name in enumerate(unique_sppgs):
        nodes, dist_matrix, points = cluster_data[sppg_name]
        # Run with higher budget for final map
        route, d = run_ga(nodes, dist_matrix, best_scen['sel'], best_scen['cross'], best_scen['mut'], pop_size=50, generations=100)
        c = colors[idx % len(colors)]
        group = folium.FeatureGroup(name=f"{sppg_name[-20:]} ({round(d/1000, 2)} km)")
        path = get_osrm_geometry([(points.iloc[i]["lat"], points.iloc[i]["lng"]) for i in route])
        folium.PolyLine(path, color=c, weight=5, opacity=0.8).add_to(group)
        for i, pt_idx in enumerate(route[:-1]):
            row = points.iloc[pt_idx]
            ic = 'industry' if row["tipe"] == "DEPOT" else 'graduation-cap'
            folium.Marker([row["lat"], row["lng"]], icon=folium.Icon(color='black' if row["tipe"] == "DEPOT" else c, icon=ic, prefix='fa')).add_to(group)
        group.add_to(m)

    folium.LayerControl().add_to(m)
    m.save("final_comparison_ga_map.html")
    print("\n[*] Map updated: final_comparison_ga_map.html")
