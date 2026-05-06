import pandas as pd
import random
import folium
import requests
import json
import os
from copy import deepcopy

#  LOAD DATA & MATRICES (SYNC WITH OSRM)
def load_all_data():
    print("[*] Loading clustered data and OSRM matrices...")
    df_schools = pd.read_csv("sekolah_terklaster.csv")
    df_sppg = pd.read_csv("sppg_sukolilo.csv")
    
    matrix_folder = "matriks_jarak"
    cluster_data = {}
    
    if not os.path.exists(matrix_folder):
        print(f"❌ Folder {matrix_folder} tidak ditemukan!")
        return {}

    json_files = [f for f in os.listdir(matrix_folder) if f.endswith(".json")]
    for f in json_files:
        with open(os.path.join(matrix_folder, f), "r") as jfile:
            data = json.load(jfile)
            sppg_name = data["sppg_pusat"]
            df_cluster = df_schools[df_schools["sppg_terdekat"] == sppg_name]
            depot_row = df_sppg[df_sppg["nama"] == sppg_name].iloc[0]
            points_info = [{"nama": depot_row["nama"], "lat": depot_row["lat"], "lng": depot_row["lng"], "tipe": "DEPOT"}]
            node_names = data["node_names"]
            for i in range(1, len(node_names)):
                name = node_names[i]
                school_match = df_cluster[df_cluster["nama_sekolah"] == name]
                if not school_match.empty:
                    school_row = school_match.iloc[0]
                    points_info.append({
                        "nama": name,
                        "lat": school_row["lat"],
                        "lng": school_row["lng"],
                        "tipe": "SEKOLAH"
                    })
                else:
                    print(f"⚠️ Warning: {name} tidak ditemukan di CSV!")
            
            cluster_data[sppg_name] = {
                "dist_matrix": data["distance_matrix_meters"],
                "points": pd.DataFrame(points_info),
                "num_nodes": data["jumlah_node"]
            }
            
    return cluster_data

# GA OPERATORS
def calculate_fitness(route, dist_matrix):
    total = dist_matrix[0][route[0]] 
    for i in range(len(route) - 1):
        total += dist_matrix[route[i]][route[i + 1]]
    total += dist_matrix[route[-1]][0] 
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
def run_ga(num_nodes, dist_matrix, sel_op, cross_op, mut_op, pop_size=30, generations=100, mut_rate=0.2, init_pop=None):
    if num_nodes <= 1: return [0, 0], 0, [0]
    base = list(range(1, num_nodes))
    pop = init_pop if init_pop else [random.sample(base, len(base)) for _ in range(pop_size)]
    
    history = []
    best = min(pop, key=lambda x: calculate_fitness(x, dist_matrix))
    
    for _ in range(generations):
        new_pop = [best]
        while len(new_pop) < pop_size:
            p1, p2 = sel_op(pop, dist_matrix), sel_op(pop, dist_matrix)
            child = mut_op(cross_op(p1, p2), mut_rate)
            new_pop.append(child)
        pop = new_pop
        curr_best = min(pop, key=lambda x: calculate_fitness(x, dist_matrix))
        if calculate_fitness(curr_best, dist_matrix) < calculate_fitness(best, dist_matrix):
            best = curr_best
        history.append(calculate_fitness(best, dist_matrix))
        
    return [0] + best + [0], calculate_fitness(best, dist_matrix), history

# OSRM GEOMETRY
def get_osrm_geometry(coords):
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
    try:
        r = requests.get(f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson", timeout=10)
        return [(lat, lon) for lon, lat in r.json()["routes"][0]["geometry"]["coordinates"]]
    except: return coords

# GRAFIK KONVERGENSI (HTML + Chart.js)
def simpan_grafik_konvergensi_ga(semua_history, nama_sppg_list, rekap_hasil, best_scen_name):
    max_len = max(len(h) for h in semua_history) if semua_history else 100
    labels_js = json.dumps(list(range(1, max_len + 1)))
    warna_list = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6","#1ABC9C","#E67E22"]

    datasets = []
    for i, (history, nama) in enumerate(zip(semua_history, nama_sppg_list)):
        history_km = [round(d / 1000, 3) for d in history]
        while len(history_km) < max_len:
            history_km.append(history_km[-1])
        warna = warna_list[i % len(warna_list)]
        datasets.append(f"""{{
            label: "{nama}",
            data: {json.dumps(history_km)},
            borderColor: "{warna}",
            backgroundColor: "{warna}22",
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
            tension: 0.3
        }}""")

    tabel_rows = ""
    for r in rekap_hasil:
        tabel_rows += f"""
        <tr>
            <td>{r['sppg']}</td>
            <td style="text-align:center">{r['jumlah_sekolah']}</td>
            <td style="text-align:center">{r['jarak_rute_km']} km</td>
        </tr>"""

    total_km = sum(r["jarak_rute_km"] for r in rekap_hasil)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Konvergensi GA - MBG Sukolilo</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }}
        .container {{ background: white; padding: 24px; border-radius: 10px;
                      box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 960px; margin: auto; }}
        h2 {{ color: #333; margin-top: 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }}
        th {{ background: #f0f0f0; padding: 8px 12px; text-align: left; border-bottom: 2px solid #ddd; }}
        td {{ padding: 7px 12px; border-bottom: 1px solid #eee; }}
        tr:hover td {{ background: #f9f9f9; }}
        .total {{ font-weight: bold; background: #f0f0f0; }}
    </style>
</head>
<body>
<div class="container">
    <h2>🧬 Konvergensi GA — Distribusi MBG Kecamatan Sukolilo</h2>
    <p style="color:#666">Best Scenario: <b>{best_scen_name}</b> | Sumbu X: Generasi | Sumbu Y: Jarak terbaik (km)</p>
    <canvas id="chart" height="380"></canvas>

    <h3 style="margin-top:28px">📊 Rekap Hasil per SPPG</h3>
    <table>
        <tr>
            <th>SPPG</th>
            <th style="text-align:center">Sekolah</th>
            <th style="text-align:center">Jarak Rute</th>
        </tr>
        {tabel_rows}
        <tr class="total">
            <td><b>TOTAL</b></td>
            <td style="text-align:center"><b>{sum(r['jumlah_sekolah'] for r in rekap_hasil)}</b></td>
            <td style="text-align:center"><b>{total_km:.2f} km</b></td>
        </tr>
    </table>
</div>

<script>
new Chart(document.getElementById('chart').getContext('2d'), {{
    type: 'line',
    data: {{
        labels: {labels_js},
        datasets: [{",".join(datasets)}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{ position: 'bottom', labels: {{ boxWidth: 12, font: {{ size: 11 }} }} }},
            title: {{ display: true, text: 'Konvergensi Genetic Algorithm per Klaster SPPG' }}
        }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Generasi' }} }},
            y: {{ title: {{ display: true, text: 'Jarak Terbaik (km)' }} }}
        }}
    }}
}});
</script>
</body>
</html>"""

    with open("Grafik_Konvergensi_GA.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("📊 Grafik konvergensi disimpan: Grafik_Konvergensi_GA.html")

# MAIN
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    cluster_data = load_all_data()
    if not cluster_data:
        exit(1)
        
    unique_sppgs = sorted(cluster_data.keys())

    scenarios = [
        {"name": "Tourn + OX1 + Inv", "sel": selection_tournament, "cross": crossover_ox1, "mut": mutation_inversion},
        # {"name": "Roul + OX1 + Inv", "sel": selection_roulette, "cross": crossover_ox1, "mut": mutation_inversion},
        # {"name": "SUS + OX1 + Inv", "sel": selection_sus, "cross": crossover_ox1, "mut": mutation_inversion},
        # {"name": "Tourn + OX1 + Swap", "sel": selection_tournament, "cross": crossover_ox1, "mut": mutation_swap},
        # {"name": "Roul + OX1 + Swap", "sel": selection_roulette, "cross": crossover_ox1, "mut": mutation_swap},
        # {"name": "SUS + OX1 + Swap", "sel": selection_sus, "cross": crossover_ox1, "mut": mutation_swap},
        # {"name": "Tourn + OX1 + Scramble", "sel": selection_tournament, "cross": crossover_ox1, "mut": mutation_scramble},
        # {"name": "Roul + OX1 + Scramble", "sel": selection_roulette, "cross": crossover_ox1, "mut": mutation_scramble},
        # {"name": "SUS + OX1 + Scramble", "sel": selection_sus, "cross": crossover_ox1, "mut": mutation_scramble},
    ]

    # Pre-generate populations for fairness
    initial_populations = {}
    random.seed(42)
    for s in unique_sppgs:
        n_nodes = cluster_data[s]["num_nodes"]
        base = list(range(1, n_nodes))
        initial_populations[s] = [random.sample(base, len(base)) for _ in range(30)]

    all_performance = []
    print("\n" + "="*70)
    print("GA PERFORMANCE COMPARISON")
    print("="*70)

    for scen in scenarios:
        for sppg_name in unique_sppgs:
            data = cluster_data[sppg_name]
            _, dist, _ = run_ga(data["num_nodes"], data["dist_matrix"], scen['sel'], scen['cross'], scen['mut'], 
                             pop_size=50, generations=100, init_pop=deepcopy(initial_populations[sppg_name]))
            all_performance.append({"Scenario": scen['name'], "Cluster": sppg_name[-20:], "Dist (km)": round(dist/1000, 2)})

    df_perf = pd.DataFrame(all_performance)
    print(df_perf.pivot(index="Cluster", columns="Scenario", values="Dist (km)"))

    summary = df_perf.groupby("Scenario")["Dist (km)"].sum().reset_index().sort_values("Dist (km)")
    print("\n" + "="*70)
    print("RANKING SKENARIO (TOTAL JARAK)")
    print("="*70)
    print(summary)

    best_scen_name = summary.iloc[0]["Scenario"]
    best_scen = next(s for s in scenarios if s["name"] == best_scen_name)
    
    print(f"\n[*] Generating final map for best scenario: {best_scen_name}")
    m = folium.Map(location=[-7.2991, 112.7838], zoom_start=14)
    colors = ["blue","green","purple","orange","darkred","cadetblue","black"]
    
    total_km = 0
    total_sekolah = 0
    
    semua_history = []
    nama_sppg_list = []
    rekap_hasil = []

    for idx, sppg_name in enumerate(unique_sppgs):
        data = cluster_data[sppg_name]
        # Higher budget for final visualization
        route, d, history = run_ga(data["num_nodes"], data["dist_matrix"], best_scen['sel'], best_scen['cross'], best_scen['mut'], 
                          pop_size=50, generations=200)
        
        dist_km = round(d/1000, 2)
        total_km += dist_km
        total_sekolah += (data["num_nodes"] - 1)
        
        semua_history.append(history)
        nama_sppg_list.append(sppg_name)
        rekap_hasil.append({
            "sppg": sppg_name,
            "jumlah_sekolah": data["num_nodes"] - 1,
            "jarak_rute_km": dist_km
        })
        
        c = colors[idx % len(colors)]
        group = folium.FeatureGroup(name=f"Rute {sppg_name} ({dist_km} km)")
        
        # Geometry sequence
        route_coords = [(data["points"].iloc[i]["lat"], data["points"].iloc[i]["lng"]) for i in route]
        path = get_osrm_geometry(route_coords)
        folium.PolyLine(path, color=c, weight=5, opacity=0.8, tooltip=f"{sppg_name} — {dist_km} km").add_to(group)
        
        for i, pt_idx in enumerate(route[:-1]):
            row = data["points"].iloc[pt_idx]
            if row["tipe"] == "DEPOT":
                folium.Marker(
                    [row["lat"], row["lng"]],
                    popup=folium.Popup(
                        f"<b>🏭 Dapur SPPG</b><br>{row['nama']}<br>"
                        f"Melayani: {data['num_nodes']-1} sekolah<br>"
                        f"Jarak rute: {dist_km} km<br>"
                        f"Scenario: {best_scen_name}",
                        max_width=280
                    ),
                    tooltip=f"🏭 {row['nama']}",
                    icon=folium.Icon(color="red", icon="cutlery", prefix="fa")
                ).add_to(group)
            else:
                folium.Marker(
                    [row["lat"], row["lng"]],
                    popup=folium.Popup(
                        f"<b>🏫 {row['nama']}</b><br>Kunjungan ke-{i}",
                        max_width=200
                    ),
                    tooltip=f"Ke-{i}: {row['nama']}",
                    icon=folium.Icon(color=c, icon="book")
                ).add_to(group)
        group.add_to(m)

    # Legend/Info box
    info_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.2);
                font-family:Arial;font-size:12px">
        <b>🧬 GA — Distribusi MBG Sukolilo</b><br>
        <hr style="margin:5px 0">
        Total jarak    : <b>{total_km:.2f} km</b><br>
        Jumlah SPPG    : <b>{len(unique_sppgs)}</b><br>
        Jumlah sekolah : <b>{total_sekolah}</b><br>
        Best Scenario  : <b>{best_scen_name}</b>
    </div>"""
    
    m.get_root().html.add_child(folium.Element(info_html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save("Peta_Rute_GA_MBG_Sukolilo.html")
    print(f"\n[*] Map saved: Peta_Rute_GA_MBG_Sukolilo.html")

    # SIMPAN GRAFIK KONVERGENSI
    simpan_grafik_konvergensi_ga(semua_history, nama_sppg_list, rekap_hasil, best_scen_name)

