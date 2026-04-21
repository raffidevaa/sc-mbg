# =========================
# 1. IMPORT LIBRARY
# =========================
import pandas as pd
import random
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import folium
from folium.plugins import MiniMap

random.seed(42)  # seed agar hasil GA reproducible (sama di setiap run)

# =========================
# 2. LOAD DATA
# =========================
sppg = pd.read_csv("sppg_sukolilo.csv")
sekolah = pd.read_csv("sekolah_sukolilo.csv")

points = pd.concat([sppg, sekolah], ignore_index=True)

# pastikan numeric
points["lat"] = pd.to_numeric(points["lat"], errors="coerce")
points["lng"] = pd.to_numeric(points["lng"], errors="coerce")

# buang yang null
points = points.dropna(subset=["lat", "lng"])

# =========================
# 3. LOAD OSM GRAPH
# =========================
# place_name = "Sukolilo, Surabaya, Indonesia"
# G = ox.graph_from_place(place_name, network_type="drive")

# =========================
# 3. LOAD OSM GRAPH (IMPROVED)
# =========================
place_name = "Sukolilo, Surabaya, Indonesia"
G = ox.graph_from_place(place_name, network_type="drive")

# Ambil komponen terbesar yang saling terhubung sempurna
G = ox.truncate.largest_component(G, strongly=True)

# =========================
# 4. MAP KE NODE OSM
# =========================
def get_node(lat, lng):
    return ox.distance.nearest_nodes(G, lng, lat)

points["node"] = points.apply(lambda row: get_node(row["lat"], row["lng"]), axis=1)
nodes = points["node"].tolist()

# =========================
# 5. DISTANCE MATRIX (ROAD BASED)
# =========================
dist_matrix = {}

print("Computing distance matrix...")

for i in range(len(nodes)):
    for j in range(len(nodes)):
        if i == j:
            dist_matrix[(i, j)] = 0
        else:
            try:
                dist = nx.shortest_path_length(
                    G, nodes[i], nodes[j], weight="length"
                )
            except nx.NetworkXNoPath:
                dist = 1e9  # penalti besar

            dist_matrix[(i, j)] = dist

# =========================
# 6. FITNESS FUNCTION
# =========================
def fitness(route):
    total = 0

    for i in range(len(route) - 1):
        total += dist_matrix[(route[i], route[i + 1])]

    # kembali ke depot
    total += dist_matrix[(route[-1], route[0])]

    return total

# =========================
# 7. POPULATION
# =========================
def create_population(size):
    population = []
    base = list(range(1, len(nodes)))  # tanpa depot

    for _ in range(size):
        individual = base[:]
        random.shuffle(individual)
        individual = [0] + individual  # depot di depan
        population.append(individual)

    return population

# =========================
# 8. SELECTION
# =========================
def selection(pop):
    return min(random.sample(pop, 3), key=fitness)

# =========================
# 9. CROSSOVER
# =========================
def crossover(p1, p2):
    start, end = sorted(random.sample(range(1, len(p1)), 2))

    child = [None] * len(p1)
    child[start:end] = p1[start:end]

    fill = [x for x in p2 if x not in child]

    idx = 1
    for i in range(len(child)):
        if child[i] is None:
            child[i] = fill.pop(0)

    child[0] = 0
    return child

# =========================
# 10. MUTATION
# =========================
def mutate(route, rate=0.1):
    route = route[:]
    if random.random() < rate:
        i, j = random.sample(range(1, len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# =========================
# 11. GENETIC ALGORITHM
# =========================
def genetic_algorithm(pop_size=20, generations=50):
    pop = create_population(pop_size)
    best = min(pop, key=fitness)

    history = []

    for gen in range(generations):
        new_pop = []

        for _ in range(pop_size):
            p1 = selection(pop)
            p2 = selection(pop)

            child = crossover(p1, p2)
            child = mutate(child)

            new_pop.append(child)

        pop = new_pop

        current_best = min(pop, key=fitness)

        if fitness(current_best) < fitness(best):
            best = current_best

        history.append(fitness(best))
        print(f"Gen {gen+1}: Best Distance = {fitness(best):.2f}")

    return best, history

# =========================
# 12. RUN GA
# =========================
best_route, history = genetic_algorithm()

print("\nBest Route:", best_route)
print("Best Distance:", fitness(best_route))

# # =========================
# # 13. PLOT ROUTE (OSM)
# # =========================
# route_nodes = []

# for i in range(len(best_route) - 1):
#     path = nx.shortest_path(
#         G,
#         nodes[best_route[i]],
#         nodes[best_route[i + 1]],
#         weight="length"
#     )
#     route_nodes.extend(path)

# # kembali ke depot
# path = nx.shortest_path(
#     G,
#     nodes[best_route[-1]],
#     nodes[best_route[0]],
#     weight="length"
# )
# route_nodes.extend(path)

# fig, ax = ox.plot_graph_route(G, route_nodes, node_size=0)
# plt.savefig("rute_mbg.png", dpi=300)
# plt.show()

# =========================
# 13. PLOT ROUTE (OSM) - FIXED
# =========================
# route_nodes = []

# print("Preparing route for visualization...")
# for i in range(len(best_route)):
#     start_node = nodes[best_route[i]]
#     # Jika sudah di akhir, kembali ke depot (indeks 0)
#     end_node = nodes[best_route[(i + 1) % len(best_route)]]
    
#     try:
#         path = nx.shortest_path(G, start_node, end_node, weight="length")
#         route_nodes.extend(path)
#     except nx.NetworkXNoPath:
#         print(f"Warning: No path between node {start_node} and {end_node}. Skipping segment.")
#         continue

# if route_nodes:
#     fig, ax = ox.plot_graph_route(G, route_nodes, node_size=0, route_color='r', route_linewidth=3)
#     plt.savefig("rute_mbg.png", dpi=300)
#     plt.show()
# else:
#     print("Error: No valid route segments found to plot.")

# =========================
# 13. PLOT ROUTE — FOLIUM INTERACTIVE MAP
# =========================
all_routes = []

print("Preparing route segments for visualization...")
for i in range(len(best_route)):
    start_node = nodes[best_route[i]]
    end_node   = nodes[best_route[(i + 1) % len(best_route)]]
    try:
        path = nx.shortest_path(G, start_node, end_node, weight="length")
        all_routes.append(path)
    except nx.NetworkXNoPath:
        print(f"Warning: No path between {start_node} and {end_node}. Segment hidden.")

if all_routes:
    # ============================================
    # 13. FOLIUM INTERACTIVE MAP
    # ============================================

    # Helper: ambil nama lokasi dari dataframe
    def get_label(idx):
        row = points.iloc[idx]
        name = row.get('nama_sekolah', None)
        if pd.isna(name) or name is None:
            name = row.get('nama', f'Lokasi {idx}')
        jenjang = row.get('jenjang', '')
        alamat  = row.get('alamat', '-')
        if pd.isna(jenjang): jenjang = 'Depot'
        if pd.isna(alamat):  alamat  = '-'
        return str(name), str(jenjang), str(alamat)

    # Pusat peta
    center_lat = points["lat"].mean()
    center_lng = points["lng"].mean()

    # Buat peta Folium dengan tile CartoDB Voyager
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=13,
        tiles='CartoDB Voyager'
    )

    # --- Gambar segmen rute ---
    for path in all_routes:
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]  # (lat, lng)
        folium.PolyLine(
            coords,
            color='#E63946',
            weight=4,
            opacity=0.85,
            tooltip='Rute Optimal GA'
        ).add_to(m)

    # --- Marker sekolah ---
    for order, idx in enumerate(best_route[1:], start=1):
        node = nodes[idx]
        lat  = G.nodes[node]['y']
        lng  = G.nodes[node]['x']
        name, jenjang, alamat = get_label(idx)

        popup_html = f"""
        <div style="font-family:Arial;font-size:13px;min-width:200px">
            <b style="color:#1D3557">#{order} — {name}</b><br>
            <span style="color:#555">Jenjang:</span> {jenjang}<br>
            <span style="color:#555">Alamat:</span> {alamat}
        </div>"""

        folium.CircleMarker(
            location=[lat, lng],
            radius=7,
            color='white',
            weight=1.5,
            fill=True,
            fill_color='#1D6FA4',
            fill_opacity=0.9,
            tooltip=f'#{order} {name}',
            popup=folium.Popup(popup_html, max_width=260)
        ).add_to(m)

    # --- Marker depot ---
    depot_node = nodes[best_route[0]]
    depot_lat  = G.nodes[depot_node]['y']
    depot_lng  = G.nodes[depot_node]['x']
    depot_name, _, depot_alamat = get_label(best_route[0])

    depot_popup = f"""
    <div style="font-family:Arial;font-size:13px;min-width:200px">
        <b style="color:#E63946">⭐ DEPOT — {depot_name}</b><br>
        <span style="color:#555">Alamat:</span> {depot_alamat}
    </div>"""

    folium.Marker(
        location=[depot_lat, depot_lng],
        tooltip='⭐ Depot SPPG (Titik Awal/Akhir)',
        popup=folium.Popup(depot_popup, max_width=260),
        icon=folium.Icon(color='orange', icon='star', prefix='fa')
    ).add_to(m)

    # --- MiniMap di pojok ---
    MiniMap(toggle_display=True).add_to(m)

    # --- Title overlay ---
    title_html = f'''
    <div style="position:fixed;top:12px;left:55px;z-index:9999;
                background:rgba(255,255,255,0.92);padding:10px 16px;
                border-radius:8px;border:1px solid #ccc;
                font-family:Arial;font-size:14px;color:#1D3557;">
        <b>🗺️ Rute Optimal MBG Sukolilo</b><br>
        <span style="font-size:12px;color:#555">
        Best Distance: <b>{fitness(best_route)/1000:.2f} km</b> &nbsp;|&nbsp;
        {len(best_route)-1} Sekolah &nbsp;|&nbsp; 50 Generasi
        </span>
    </div>'''
    m.get_root().html.add_child(folium.Element(title_html))

    # Simpan
    m.save("rute_mbg.html")
    print("Peta interaktif disimpan: rute_mbg.html")

# =========================
# 14. PLOT KONVERGENSI (OPSIONAL BAGUS)
# =========================
plt.figure()
plt.plot(history)
plt.xlabel("Generation")
plt.ylabel("Distance")
plt.title("GA Convergence")
plt.savefig("konvergensi.png", dpi=300)
plt.show()
