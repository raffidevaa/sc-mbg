import pandas as pd
import numpy as np
import json
import folium
import os
import random
import math
import requests
import optuna

# ==========================================
# KONFIGURASI
# ==========================================
MODE_TUNING = True

DEFAULT_T_INIT = 100
DEFAULT_ALPHA  = 0.995
DEFAULT_T_MIN  = 1

# ==========================================
# CLASS SA
# ==========================================
class SimulatedAnnealingTSP:
    def __init__(self, distance_matrix, T_init, alpha, T_min):
        self.distances = np.array(distance_matrix)
        self.n = len(distance_matrix)
        self.T_init = T_init
        self.alpha = alpha
        self.T_min = T_min

    def fitness(self, route):
        total = 0
        for i in range(len(route) - 1):
            total += self.distances[route[i]][route[i+1]]
        total += self.distances[route[-1]][route[0]]
        return total

    def initial_solution(self):
        route = list(range(1, self.n))
        random.shuffle(route)
        return [0] + route

    def get_neighbor(self, route):
        new_route = route[:]
        i, j = sorted(random.sample(range(1, len(route)), 2))
        new_route[i:j] = reversed(new_route[i:j])
        return new_route

    def run(self):
        current = self.initial_solution()
        best = current[:]

        T = self.T_init
        history = []

        while T > self.T_min:
            neighbor = self.get_neighbor(current)
            delta = self.fitness(neighbor) - self.fitness(current)

            if delta < 0 or random.random() < math.exp(-delta / T):
                current = neighbor

            if self.fitness(current) < self.fitness(best):
                best = current

            history.append(self.fitness(best))
            T *= self.alpha

        return best, self.fitness(best), history

# ==========================================
# OPTUNA TUNING (FIX CLEAN OUTPUT)
# ==========================================
def optuna_tuning(matrix):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        T_init = trial.suggest_categorical("T_init", [100, 500, 1000])
        alpha  = trial.suggest_categorical("alpha", [0.99, 0.995, 0.999])
        T_min  = 1

        sa = SimulatedAnnealingTSP(matrix, T_init, alpha, T_min)
        _, best_dist, _ = sa.run()
        return best_dist

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    p = study.best_params
    return p["T_init"], p["alpha"], 1

def dapatkan_jalur_aspal_osrm(coords):
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
    url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson"

    try:
        data = requests.get(url, timeout=10).json()
        if data.get("code") == "Ok":
            return [(lat, lon) for lon, lat in data["routes"][0]["geometry"]["coordinates"]]
    except:
        pass

    return coords

# ==========================================
# MAIN
# ==========================================
def main():
    folder = "matriks_jarak"
    df_master = pd.read_csv("sekolah_terklaster.csv")
    df_sppg = pd.read_csv("sppg_sukolilo.csv", sep=None, engine='python')

    m = folium.Map(location=[-7.29,112.78], zoom_start=13)
    colors = [
        "red","blue","purple","darkgreen","black",
        "cadetblue","darkred","darkblue","darkpurple","orange"  
    ]   

    folium.Element("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """).add_to(m)

    histories = []
    names = []
    total_all = 0

    table_rows = ""
    for file in os.listdir(folder):
        if not file.endswith(".json"):
            continue

        data = json.load(open(os.path.join(folder, file)))
        nama = data["sppg_pusat"]
        matrix = data["distance_matrix_meters"]

        print(f"\nProcessing cluster: {nama}")

        # ===== TUNING =====
        if MODE_TUNING:
            T, alpha, Tmin = optuna_tuning(matrix)
        else:
            T, alpha, Tmin = DEFAULT_T_INIT, DEFAULT_ALPHA, DEFAULT_T_MIN

        # ===== RUN SA =====
        sa = SimulatedAnnealingTSP(matrix, T, alpha, Tmin)
        best_route, best_distance, history = sa.run()

        histories.append(history)
        names.append(nama)

        km = round(best_distance / 1000, 2)
        total_all += best_distance

        jumlah_sekolah = len(df_master[df_master['sppg_terdekat'] == nama])

        table_rows += f"""
        <tr>
            <td>{nama}</td>
            <td>{jumlah_sekolah}</td>
            <td>{km}</td>
            <td>{T}</td>
            <td>{alpha}</td>
        </tr>
            """

        print(f"Best Parameter -> T_INIT: {T}, ALPHA: {alpha}")
        print(f"Total Distance : {km} km")

        df_klaster = df_master[df_master['sppg_terdekat'] == nama].reset_index(drop=True)
        baris_sppg = df_sppg[df_sppg['nama'] == nama].iloc[0]

        nama_lokasi = [nama] + df_klaster['nama_sekolah'].tolist()

        print("Route Sequence :")
        for i, idx_node in enumerate(best_route):
            print(f"   {i+1}. {nama_lokasi[idx_node]}")

        # =========================
        # KOORDINAT
        # =========================
        koordinat = [(baris_sppg['lat'], baris_sppg['lng'])]
        for _, row in df_klaster.iterrows():
            koordinat.append((row['lat'], row['lng']))

        # =========================
        # ROUTE KE KOORDINAT
        # =========================
        titik = [koordinat[i] for i in best_route]
        titik.append(titik[0])  # kembali ke depot

        jalur = dapatkan_jalur_aspal_osrm(titik)

        # =========================
        # PLOT KE MAP
        # =========================
        group = folium.FeatureGroup(name=nama)
        
        warna = colors[len(names) % len(colors)]

        for i, idx_node in enumerate(best_route):
            coord = koordinat[idx_node]
            nama_node = nama_lokasi[idx_node]

            if idx_node == 0:
                folium.Marker(
                coord,
                popup=f"Depot: {nama_node}",
                icon=folium.Icon(color='red', icon='cutlery', prefix='fa')
            ).add_to(group)
            else:
                folium.Marker(
                    coord,
                    popup=f"{i}. {nama_node}",
                    icon=folium.Icon(color=warna)
                ).add_to(group)

        folium.PolyLine(jalur, color=warna, weight=5).add_to(group)
        group.add_to(m)

    print(f"\nTotal Jarak Seluruh SPPG : {round(total_all/1000,2)} km")
    total_km = round(total_all/1000, 2)

    info_html = f"""
    <div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    z-index: 1000;
    background: white;
    padding: 12px 16px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    font-family: Arial;
    font-size: 12px;
    ">
    <b>🔥 SA — Distribusi MBG Sukolilo</b><br>
    <hr style="margin:5px 0">
    Total jarak    : <b>{total_km} km</b><br>
    Jumlah SPPG    : <b>{len(names)}</b><br>
    Jumlah sekolah : <b>{sum(len(df_master[df_master['sppg_terdekat']==n]) for n in names)}</b><br>
    Mode           : <b>{"Optuna" if MODE_TUNING else "Default"}</b>
    </div>
    """

    m.get_root().html.add_child(folium.Element(info_html))

    folium.LayerControl(collapsed=False).add_to(m)
    m.save("Peta_Rute_SA_MBG_Sukolilo.html")

    print("Map saved as Peta_Rute_SA_MBG_Sukolilo.html")

    # =========================
    # GENERATE DASHBOARD SA
    # =========================
    total_km = round(total_all/1000, 2)
    total_sekolah = sum(len(df_master[df_master['sppg_terdekat']==n]) for n in names)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analisis SA MBG Sukolilo</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{
                font-family: Arial;
                background: #f5f6fa;
                padding: 20px;
            }}
            .card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .grid {{
                display: flex;
                gap: 20px;
            }}
            .box {{
                flex: 1;
                background: #ecf0f1;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid black;
                padding: 8px;
                text-align: center;
            }}
            table th {{
                background: #f0f0f0;
            }}
        </style>
    </head>

    <body>

    <div class="card">
        <h2>🔥 Analisis Simulated Annealing — Distribusi MBG Sukolilo</h2>

        <div class="grid">
            <div class="box">
                <h3>{total_km} km</h3>
                <p>Total jarak semua SPPG</p>
            </div>

            <div class="box">
                <h3>{len(names)}</h3>
                <p>Jumlah SPPG</p>
            </div>

            <div class="box">
                <h3>{total_sekolah}</h3>
                <p>Total sekolah dilayani</p>
            </div>
        </div>
    </div>

    <div class="card">
        <h3>Grafik Konvergensi SA</h3>
        <canvas id="chart"></canvas>
    </div>

    <div class="card">
        <h3>📋 Rekap Hasil per SPPG</h3>
        <table>
            <tr>
                <th>SPPG</th>
                <th>Jumlah Sekolah</th>
                <th>Jarak (km)</th>
                <th>T_init</th>
                <th>Alpha</th>
            </tr>
            {table_rows}
            <tr style="font-weight:bold; background:#dfe6e9;">
                <td>TOTAL</td>
                <td>{total_sekolah}</td>
                <td>{total_km}</td>
                <td>-</td>
                <td>-</td>
            </tr>
        </table>

    </div>

    <script>
    const data = {{
        labels: {list(range(1, min(70, max(len(h) for h in histories))+1))},
        datasets: [
            {",".join([
                f"""{{
                    label: "{names[i]}",
                    data: {[round(x/1000,2) for x in histories[i][:70]]},
                    fill: false,
                    tension: 0.3
                }}"""
                for i in range(len(names))
            ])}
        ]
    }};

    new Chart(document.getElementById('chart'), {{
        type: 'line',
        data: data,
        options: {{
            responsive: true,
            scales: {{
                x: {{
                    title: {{
                        display: true,
                        text: 'Iterasi'
                    }}
                }},
                y: {{
                    title: {{
                        display: true,
                        text: 'Jarak (km)'
                    }}
                }}
            }}
        }}
    }});
    </script>

    </body>
    </html>
    """

    # SAVE FILE TERPISAH
    with open("Analisis_SA_MBG_Sukolilo.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Dashboard saved as Analisis_SA_MBG_Sukolilo.html")

if __name__ == "__main__":
    main()