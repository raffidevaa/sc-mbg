import pandas as pd
import numpy as np
import json
import folium
import os
import random
import requests
import optuna

# --- Mode & Parameter PSO ---
MODE_TUNING  = True  # True: Optuna, False: pakai DEFAULT

N_PARTICLES  = 50    # jumlah partikel — lebih banyak agar eksplorasi ruang solusi lebih luas
N_ITERATIONS = 100   # jumlah iterasi
W_DAMP       = 0.995 # redaman inersia lebih lambat — eksplorasi berlangsung lebih lama

# Parameter default (dipakai jika MODE_TUNING = False)
DEFAULT_W  = 0.8   # bobot inersia
DEFAULT_C1 = 0.7   # koefisien kognitif
DEFAULT_C2 = 0.7   # koefisien sosial


class ParticleSwarmTSP:
    """
    PSO diskrit berbasis Swap Sequence untuk TSP.
    Posisi partikel = permutasi urutan kunjungan (list indeks node).
    Kecepatan partikel = daftar operasi swap (i, j).
    Update kecepatan: v(t+1) = inersia ⊕ kognitif ⊕ sosial
    """

    def __init__(self, distance_matrix, n_particles, n_iterations, w, c1, c2, w_damp):
        self.dist    = np.array(distance_matrix, dtype=float)
        np.fill_diagonal(self.dist, np.inf)
        self.n_nodes = len(distance_matrix)
        self.n_p     = n_particles
        self.n_iter  = n_iterations
        self.w       = w
        self.c1      = c1
        self.c2      = c2
        self.w_damp  = w_damp

    def fitness(self, route):
        """Total jarak rute (loop tertutup kembali ke depot)."""
        total = sum(self.dist[route[i]][route[i + 1]] for i in range(len(route) - 1))
        total += self.dist[route[-1]][route[0]]
        return total

    def get_swap_sequence(self, source, target):
        """Hasilkan urutan swap (i, j) yang mentransformasi source menjadi target."""
        a, swaps = source[:], []
        for i in range(len(a)):
            if a[i] != target[i]:
                j = a.index(target[i], i)
                swaps.append((i, j))
                a[i], a[j] = a[j], a[i]
        return swaps

    def apply_swaps(self, route, swaps):
        """Terapkan daftar swap ke rute secara berurutan."""
        r = route[:]
        for i, j in swaps:
            r[i], r[j] = r[j], r[i]
        return r

    def filter_swaps(self, swaps, prob):
        """Saring setiap swap secara acak dengan probabilitas prob (Bernoulli sampling)."""
        return [(i, j) for (i, j) in swaps if random.random() < min(1.0, prob)]

    def update_velocity(self, velocity, pos, p_best, g_best, w):
        """Hitung kecepatan baru: inersia ⊕ kognitif (arah pBest) ⊕ sosial (arah gBest)."""
        r1, r2    = random.random(), random.random()
        inertia   = self.filter_swaps(velocity, w)
        cognitive = self.filter_swaps(self.get_swap_sequence(pos, p_best), self.c1 * r1)
        social    = self.filter_swaps(self.get_swap_sequence(pos, g_best), self.c2 * r2)
        return inertia + cognitive + social

    def random_route(self):
        """Buat rute acak; indeks 0 (depot/SPPG) selalu di posisi pertama."""
        interior = list(range(1, self.n_nodes))
        random.shuffle(interior)
        return [0] + interior

    def two_opt(self, route):
        """2-opt local search: balik sub-segmen [i..j] jika menghasilkan jarak lebih pendek."""
        best, improved = route[:], True
        while improved:
            improved = False
            for i in range(1, len(best) - 1):
                for j in range(i + 1, len(best)):
                    new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    if self.fitness(new_route) < self.fitness(best):
                        best, improved = new_route, True
                        break
                if improved:
                    break
        return best

    def run(self):
        # Inisialisasi swarm: posisi acak, kecepatan kosong
        positions  = [self.random_route() for _ in range(self.n_p)]
        velocities = [[] for _ in range(self.n_p)]
        p_bests    = [p[:] for p in positions]
        fit_pbest  = [self.fitness(p) for p in p_bests]
        best_idx   = int(np.argmin(fit_pbest))
        g_best     = p_bests[best_idx][:]
        fit_gbest  = fit_pbest[best_idx]
        history    = []
        w_current  = self.w
        for _ in range(self.n_iter):
            for i in range(self.n_p):
                velocities[i] = self.update_velocity(
                    velocities[i], positions[i], p_bests[i], g_best, w_current
                )
                positions[i] = self.apply_swaps(positions[i], velocities[i])
                f = self.fitness(positions[i])
                if f < fit_pbest[i]:
                    p_bests[i], fit_pbest[i] = positions[i][:], f
                if fit_pbest[i] < fit_gbest:
                    g_best, fit_gbest = p_bests[i][:], fit_pbest[i]
            history.append(fit_gbest)
            w_current *= self.w_damp  # kurangi inersia agar eksplorasi → eksploitasi
        # 2-opt local search: perbaiki rute terbaik PSO
        g_best_refined = self.two_opt(g_best)
        return (g_best_refined, self.fitness(g_best_refined)), history


def dapatkan_jalur_aspal_osrm(koordinat_rute_urut):
    """Ambil geometri jalan nyata via OSRM. Fallback: garis lurus jika gagal/timeout."""
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in koordinat_rute_urut])
    url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == "Ok":
            print("   Rute aspal berhasil diambil dari OSRM")
            return [(lat, lon) for lon, lat in data["routes"][0]["geometry"]["coordinates"]]
        print(f"   OSRM code bukan Ok: {data.get('code')} → pakai garis lurus")
    except requests.exceptions.Timeout:
        print("   OSRM timeout → pakai garis lurus")
    except Exception as e:
        print(f"   OSRM gagal: {e} → pakai garis lurus")
    return koordinat_rute_urut


# ==========================================
# HYPERPARAMETER TUNING (OPTUNA)
# ==========================================
def optuna_tuning_pso(matriks_jarak):
    """Cari W, C1, C2 optimal pakai Bayesian Optimization (Optuna)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        w  = trial.suggest_float("w",  0.3, 0.9)
        c1 = trial.suggest_float("c1", 0.5, 2.0)
        c2 = trial.suggest_float("c2", 0.5, 2.0)

        pso = ParticleSwarmTSP(matriks_jarak, N_PARTICLES, 30, w, c1, c2, W_DAMP)
        (_, best_dist), _ = pso.run()
        return best_dist

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)

    p = study.best_params
    print(f"   🎯 Best params → W: {p['w']:.2f} | C1: {p['c1']:.2f} | C2: {p['c2']:.2f}")
    return p["w"], p["c1"], p["c2"]


def simpan_grafik_konvergensi(semua_history, nama_sppg_list):
    """Simpan grafik konvergensi PSO per klaster SPPG sebagai HTML interaktif (Chart.js)."""
    labels_js  = json.dumps(list(range(1, N_ITERATIONS + 1)))
    warna_list = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22"]
    datasets = []
    for i, (history, nama) in enumerate(zip(semua_history, nama_sppg_list)):
        warna = warna_list[i % len(warna_list)]
        datasets.append(f"""{{
            label: "{nama[-30:]}",
            data: {json.dumps([round(d / 1000, 3) for d in history])},
            borderColor: "{warna}", backgroundColor: "{warna}22",
            borderWidth: 2, pointRadius: 0, fill: false, tension: 0.3
        }}""")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Konvergensi PSO - MBG Sukolilo</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>body{{font-family:Arial,sans-serif;padding:20px;background:#f5f5f5}}h2{{color:#333}}
.container{{background:white;padding:20px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);max-width:900px;margin:auto}}</style>
</head><body><div class="container">
<h2>🐦 Konvergensi PSO — Distribusi MBG Kecamatan Sukolilo</h2>
<p>Sumbu X: Iterasi | Sumbu Y: Jarak terbaik (km)</p>
<canvas id="chart" height="400"></canvas></div>
<script>
new Chart(document.getElementById('chart').getContext('2d'),{{
    type:'line',
    data:{{labels:{labels_js},datasets:[{",".join(datasets)}]}},
    options:{{responsive:true,plugins:{{legend:{{position:'bottom'}},
    title:{{display:true,text:'Konvergensi PSO per Klaster SPPG'}}}},
    scales:{{x:{{title:{{display:true,text:'Iterasi'}}}},y:{{title:{{display:true,text:'Jarak Terbaik (km)'}}}}}}}}
}});</script></body></html>"""
    with open("Grafik_Konvergensi_PSO.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Grafik konvergensi disimpan: Grafik_Konvergensi_PSO.html")


# ==========================================
# DASHBOARD ANALISIS PSO
# ==========================================
def simpan_analisis_pso(semua_history, nama_sppg_list, rekap_hasil):
    """Simpan dashboard analisis PSO sebagai HTML interaktif (Chart.js)."""
    total_km      = sum(r["jarak_rute_km"] for r in rekap_hasil)
    total_sekolah = sum(r["jumlah_sekolah"] for r in rekap_hasil)

    max_len   = max(len(h) for h in semua_history) if semua_history else N_ITERATIONS
    labels_js = json.dumps(list(range(1, max_len + 1)))
    warna_list = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22"]

    datasets = []
    for i, (history, nama) in enumerate(zip(semua_history, nama_sppg_list)):
        history_km = [round(d / 1000, 3) for d in history]
        while len(history_km) < max_len:
            history_km.append(history_km[-1])
        datasets.append(f"""{{
            label: "{nama}",
            data: {json.dumps(history_km)},
            borderColor: "{warna_list[i % len(warna_list)]}",
            fill: false,
            tension: 0.3,
            pointRadius: 0
        }}""")

    table_rows = ""
    for r in rekap_hasil:
        table_rows += f"""
        <tr>
            <td>{r['sppg']}</td>
            <td>{r['jumlah_sekolah']}</td>
            <td>{r['jarak_rute_km']:.2f} km</td>
            <td>W={r['w']:.2f}, C1={r['c1']:.2f}, C2={r['c2']:.2f}</td>
        </tr>"""

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Analisis PSO MBG Sukolilo</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f6fa; padding: 25px; color: #2f3640; }}
        .card {{ background: white; padding: 25px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        h2, h3 {{ margin-top: 0; color: #2f3542; }}
        .grid {{ display: flex; gap: 20px; margin-bottom: 5px; }}
        .box {{ flex: 1; background: #f1f2f6; padding: 20px; border-radius: 10px; text-align: center; border-left: 5px solid #3498db; }}
        .box h3 {{ margin: 0; font-size: 24px; color: #2f3542; }}
        .box p {{ margin: 5px 0 0; color: #747d8c; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #dcdde1; }}
        th {{ background: #f8f9fa; color: #2f3542; font-weight: 600; text-transform: uppercase; font-size: 12px; letter-spacing: 0.5px; }}
        tr:hover {{ background: #f1f2f6; }}
        .total-row {{ font-weight: bold; background: #dfe4ea !important; }}
    </style>
</head>
<body>

<div class="card">
    <h2>🐦 Analisis Particle Swarm Optimization — Distribusi MBG Sukolilo</h2>
    <div class="grid">
        <div class="box">
            <h3>{total_km:.2f} km</h3>
            <p>Total Jarak Rute</p>
        </div>
        <div class="box" style="border-left-color: #2ecc71;">
            <h3>{len(nama_sppg_list)}</h3>
            <p>Jumlah SPPG</p>
        </div>
        <div class="box" style="border-left-color: #e67e22;">
            <h3>{total_sekolah}</h3>
            <p>Total Sekolah Dilayani</p>
        </div>
        <div class="box" style="border-left-color: #9b59b6;">
            <h3>{'Optuna' if MODE_TUNING else 'Default'}</h3>
            <p>Mode Tuning</p>
        </div>
    </div>
</div>

<div class="card">
    <h3>📈 Grafik Konvergensi PSO</h3>
    <p style="font-size: 13px; color: #747d8c; margin-bottom: 20px;">Menampilkan penurunan total jarak (km) terhadap jumlah iterasi.</p>
    <canvas id="chart" height="100"></canvas>
</div>

<div class="card">
    <h3>📋 Rekapitulasi Hasil per SPPG</h3>
    <table>
        <thead>
            <tr>
                <th>Pusat SPPG</th>
                <th>Sekolah</th>
                <th>Jarak Rute</th>
                <th>Parameter PSO</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
            <tr class="total-row">
                <td>TOTAL</td>
                <td>{total_sekolah}</td>
                <td>{total_km:.2f} km</td>
                <td>-</td>
            </tr>
        </tbody>
    </table>
</div>

<script>
new Chart(document.getElementById('chart'), {{
    type: 'line',
    data: {{
        labels: {labels_js},
        datasets: [{",".join(datasets)}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: true,
        plugins: {{
            legend: {{ position: 'bottom', labels: {{ usePointStyle: true, boxWidth: 10, font: {{ size: 11 }} }} }}
        }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Iterasi' }} }},
            y: {{ title: {{ display: true, text: 'Jarak (km)' }} }}
        }}
    }}
}});
</script>

</body>
</html>"""

    with open("Analisis_PSO_MBG_Sukolilo.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Dashboard analisis disimpan: Analisis_PSO_MBG_Sukolilo.html")


def main():
    folder_matrix = "matriks_jarak"
    try:
        df_master = pd.read_csv("sekolah_terklaster.csv")
        df_sppg   = pd.read_csv("sppg_sukolilo.csv", sep=None, engine="python")
    except FileNotFoundError as e:
        print(f"File tidak ditemukan: {e}")
        return
    print("MEMULAI OPTIMASI RUTE DENGAN PARTICLE SWARM OPTIMIZATION (PSO)...\n")
    daftar_warna = ["blue", "green", "purple", "orange", "darkred", "cadetblue", "black"]
    m              = folium.Map(location=[-7.2991, 112.7838], zoom_start=14)
    semua_history  = []
    nama_sppg_list = []
    rekap_hasil    = []
    for idx, file_json in enumerate(sorted([f for f in os.listdir(folder_matrix) if f.endswith(".json")])):
        with open(os.path.join(folder_matrix, file_json), "r", encoding="utf-8") as f:
            data = json.load(f)
        nama_sppg     = data["sppg_pusat"]
        matriks_jarak = data["distance_matrix_meters"]
        node_names    = data["node_names"]
        warna_rute    = daftar_warna[idx % len(daftar_warna)]
        print(f"Memproses: {nama_sppg[-35:]}")
        print(f"   Node: {len(node_names)} titik (1 SPPG + {len(node_names) - 1} sekolah)")
        # Tuning parameter
        if MODE_TUNING:
            opt_w, opt_c1, opt_c2 = optuna_tuning_pso(matriks_jarak)
        else:
            opt_w, opt_c1, opt_c2 = DEFAULT_W, DEFAULT_C1, DEFAULT_C2

        pso = ParticleSwarmTSP(matriks_jarak, N_PARTICLES, N_ITERATIONS, opt_w, opt_c1, opt_c2, W_DAMP)
        (g_best, best_distance), history = pso.run()
        urutan_kunjungan = g_best + [0]
        jarak_km         = round(best_distance / 1000, 2)
        df_klaster     = df_master[df_master["sppg_terdekat"] == nama_sppg].reset_index(drop=True)
        baris_sppg     = df_sppg[df_sppg["nama"] == nama_sppg].iloc[0]
        koordinat_rute = [(baris_sppg["lat"], baris_sppg["lng"])] + list(zip(df_klaster["lat"], df_klaster["lng"]))
        info_rute      = [nama_sppg] + df_klaster["nama_sekolah"].tolist()
        rute_nama      = [info_rute[i] for i in urutan_kunjungan]
        print(f"   Jarak Terbaik : {jarak_km} km")
        print(f"   Rute: {' -> '.join([n[:20] for n in rute_nama])}\n")
        semua_history.append(history)
        nama_sppg_list.append(nama_sppg)
        rekap_hasil.append({
            "sppg":             nama_sppg,
            "jumlah_sekolah":   len(df_klaster),
            "jarak_rute_km":    jarak_km,
            "w":                round(opt_w, 3),
            "c1":               round(opt_c1, 3),
            "c2":               round(opt_c2, 3),
            "urutan_kunjungan": " -> ".join([n[:30] for n in rute_nama]),
        })
        grup_rute  = folium.FeatureGroup(name=f"Rute {nama_sppg[-25:]} ({jarak_km} km)")
        titik_urut = []
        for urutan, node_idx in enumerate(urutan_kunjungan):
            coord, nama_lokasi = koordinat_rute[node_idx], info_rute[node_idx]
            titik_urut.append(coord)
            if urutan >= len(urutan_kunjungan) - 1:
                continue  # skip duplikat depot penutup loop
            if node_idx == 0:
                folium.Marker(
                    coord,
                    popup=folium.Popup(f"<b>Dapur SPPG</b><br>{nama_lokasi}<br>Melayani: {len(df_klaster)} sekolah<br>Total rute: {jarak_km} km", max_width=250),
                    tooltip=folium.Tooltip(f"{nama_lokasi[-25:]}", permanent=False),
                    icon=folium.Icon(color="red", icon="cutlery", prefix="fa"),
                ).add_to(grup_rute)
            else:
                folium.Marker(
                    coord,
                    popup=folium.Popup(f"<b>{nama_lokasi}</b><br>Kunjungan ke-{urutan}", max_width=200),
                    tooltip=f"Ke-{urutan}: {nama_lokasi[:30]}",
                    icon=folium.Icon(color=warna_rute, icon="book"),
                ).add_to(grup_rute)
        folium.PolyLine(
            dapatkan_jalur_aspal_osrm(titik_urut),
            color=warna_rute, weight=5, opacity=0.8,
            tooltip=f"{nama_sppg[-25:]} — {jarak_km} km",
        ).add_to(grup_rute)
        grup_rute.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    total_km = sum(r["jarak_rute_km"] for r in rekap_hasil)
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;'
        f'padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.2);font-family:Arial;font-size:12px">'
        f'<b>🐦 PSO — Distribusi MBG Sukolilo</b><br><hr style="margin:5px 0">'
        f'Total jarak : <b>{total_km:.2f} km</b><br>'
        f'Jumlah SPPG : <b>{len(rekap_hasil)}</b><br>'
        f'Jumlah sekolah : <b>{sum(r["jumlah_sekolah"] for r in rekap_hasil)}</b></div>'
    ))
    nama_peta = "Peta_Rute_PSO_MBG_Sukolilo.html"
    m.save(nama_peta)
    pd.DataFrame(rekap_hasil).to_csv("Rekap_PSO_MBG_Sukolilo.csv", index=False, encoding="utf-8-sig")
    simpan_grafik_konvergensi(semua_history, nama_sppg_list)
    simpan_analisis_pso(semua_history, nama_sppg_list, rekap_hasil)
    print("SEMUA KLASTER BERHASIL DIPROSES!")
    print(f"Peta        : {nama_peta}")
    print(f"Konvergensi : Grafik_Konvergensi_PSO.html")
    print(f"Analisis    : Analisis_PSO_MBG_Sukolilo.html")
    print(f"Rekap CSV   : Rekap_PSO_MBG_Sukolilo.csv")
    print(f"Mode        : {'Optuna' if MODE_TUNING else 'Default'}")
    print(f"Total jarak : {total_km:.2f} km")


if __name__ == "__main__":
    main()
