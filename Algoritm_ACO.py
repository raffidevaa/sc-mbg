import pandas as pd
import numpy as np
import json
import folium
import os
import requests
import optuna

# ==========================================
# KONFIGURASI PARAMETER GLOBAL
# ==========================================
MODE_TUNING  = True  # True: Optuna, False: pakai DEFAULT
N_ANTS       = 5     # jumlah semut per iterasi
N_ITERATIONS = 100   # batas maksimum iterasi
N_BEST       = 1     # hanya 1 semut terbaik yang deposit pheromone
PATIENCE     = 15    # iterasi tanpa perbaikan sebelum early stopping

# Parameter default (dipakai jika MODE_TUNING = False)
DEFAULT_ALPHA = 0.5
DEFAULT_BETA  = 1.0
DEFAULT_DECAY = 0.3


# ==========================================
# CLASS ANT COLONY TSP
# ==========================================
class AntColonyTSP:
    def __init__(self, distance_matrix, n_ants, n_iterations, decay, alpha, beta, n_best=1):
        # FIX: default n_best=1 (bukan 5)
        self.distances    = np.array(distance_matrix, dtype=float)
        np.fill_diagonal(self.distances, np.inf)

        self.pheromone    = np.ones(self.distances.shape) / len(distance_matrix)
        self.all_inds     = range(len(distance_matrix))
        self.n_ants       = n_ants
        self.n_iterations = n_iterations
        self.decay        = decay
        self.alpha        = alpha
        self.beta         = beta
        self.n_best       = n_best

    def run(self):
        all_time_shortest = (None, np.inf)
        history           = []
        no_improve_count  = 0

        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()

            # FIX: evaporasi DULU baru deposit
            self.pheromone *= (1 - self.decay)

            # FIX: hanya N_BEST semut terbaik yang deposit
            sorted_paths = sorted(all_paths, key=lambda x: x[1])
            self.spread_pheromone(sorted_paths[:self.n_best])

            # Update solusi terbaik sepanjang waktu
            best_this_iter = sorted_paths[0]
            if best_this_iter[1] < all_time_shortest[1]:
                all_time_shortest = best_this_iter
                no_improve_count  = 0   # reset karena ada perbaikan
            else:
                no_improve_count += 1   # tidak ada perbaikan

            history.append(all_time_shortest[1])

            # Early stopping
            if no_improve_count >= PATIENCE:
                break

        return all_time_shortest, history

    def spread_pheromone(self, best_paths):
        for path, dist in best_paths:
            for move in path:
                self.pheromone[move] += 1.0 / dist

    def gen_all_paths(self):
        return [
            (path := self.gen_path(0), self.path_dist(path))
            for _ in range(self.n_ants)
        ]

    def gen_path(self, start):
        path    = []
        visited = {start}
        prev    = start

        for _ in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)

        path.append((prev, start))  # kembali ke depot
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        with np.errstate(divide="ignore", invalid="ignore"):
            heuristic = np.where(dist > 0, 1.0 / dist, 0.0)

        row   = (pheromone ** self.alpha) * (heuristic ** self.beta)
        total = row.sum()

        # FIX: hindari division by zero
        if total == 0:
            unvisited = [j for j in self.all_inds if j not in visited]
            return np.random.choice(unvisited)

        return np.random.choice(list(self.all_inds), 1, p=row / total)[0]

    def path_dist(self, path):
        return sum(self.distances[u][v] for u, v in path)


# ==========================================
# HYPERPARAMETER TUNING (OPTUNA)
# ==========================================
def optuna_tuning(matriks_jarak):
    """
    Cari alpha, beta, decay optimal pakai Bayesian Optimization.
    Setiap trial jalankan ACO singkat (5 iterasi) untuk efisiensi.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.1, 3.0)
        beta  = trial.suggest_float("beta",  1.0, 5.0)
        decay = trial.suggest_float("decay", 0.05, 0.5)

        aco = AntColonyTSP(
            matriks_jarak, N_ANTS,
            n_iterations=5, decay=decay,
            alpha=alpha, beta=beta, n_best=N_BEST
        )
        (_, best_dist), _ = aco.run()
        return best_dist

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)

    p = study.best_params
    print(f"   🎯 Best params → Alpha: {p['alpha']:.2f} | Beta: {p['beta']:.2f} | Decay: {p['decay']:.2f}")
    return p["alpha"], p["beta"], p["decay"]


# ==========================================
# OSRM — GEOMETRI JALAN NYATA
# ==========================================
def dapatkan_jalur_aspal_osrm(koordinat_rute_urut):
    """Ambil geometri jalan nyata via OSRM. Fallback ke garis lurus jika gagal."""
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in koordinat_rute_urut])
    url = (
        f"http://router.project-osrm.org/route/v1/driving/{coords_str}"
        f"?overview=full&geometries=geojson"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == "Ok":
            jalur = data["routes"][0]["geometry"]["coordinates"]
            print(f"   ✅ Rute aspal berhasil dari OSRM")
            return [(lat, lon) for lon, lat in jalur]
        print(f"   ⚠️ OSRM code: {data.get('code')} → garis lurus")
    except requests.exceptions.Timeout:
        print(f"   ⚠️ OSRM timeout → garis lurus")
    except Exception as e:
        print(f"   ⚠️ OSRM gagal: {e} → garis lurus")
    return koordinat_rute_urut


# ==========================================
# GRAFIK KONVERGENSI (HTML + Chart.js)
# ==========================================
def simpan_grafik_konvergensi(semua_history, nama_sppg_list, rekap_hasil):
    """
    Simpan grafik konvergensi ACO per SPPG sebagai HTML interaktif.
    Handle array history yang panjangnya beda karena early stopping.
    """
    max_len   = max(len(h) for h in semua_history) if semua_history else N_ITERATIONS
    labels_js = json.dumps(list(range(1, max_len + 1)))
    warna_list= ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6","#1ABC9C","#E67E22"]

    datasets = []
    for i, (history, nama) in enumerate(zip(semua_history, nama_sppg_list)):
        history_km = [round(d / 1000, 3) for d in history]
        # Extend ke nilai terakhir jika early stopping lebih awal
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

    # Tabel rekap untuk ditampilkan di grafik
    tabel_rows = ""
    for r in rekap_hasil:
        tabel_rows += f"""
        <tr>
            <td>{r['sppg']}</td>
            <td style="text-align:center">{r['jumlah_sekolah']}</td>
            <td style="text-align:center">{r['jarak_rute_km']} km</td>
            <td style="text-align:center">{r['iterasi_berhenti']}</td>
        </tr>"""

    total_km = sum(r["jarak_rute_km"] for r in rekap_hasil)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Konvergensi ACO - MBG Sukolilo</title>
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
    <h2>🐜 Konvergensi ACO — Distribusi MBG Kecamatan Sukolilo</h2>
    <p style="color:#666">Sumbu X: Iterasi | Sumbu Y: Jarak terbaik (km) | Garis datar = sudah konvergen (early stopping)</p>
    <canvas id="chart" height="380"></canvas>

    <h3 style="margin-top:28px">📊 Rekap Hasil per SPPG</h3>
    <table>
        <tr>
            <th>SPPG</th>
            <th style="text-align:center">Sekolah</th>
            <th style="text-align:center">Jarak Rute</th>
            <th style="text-align:center">Iterasi Berhenti</th>
        </tr>
        {tabel_rows}
        <tr class="total">
            <td><b>TOTAL</b></td>
            <td style="text-align:center"><b>{sum(r['jumlah_sekolah'] for r in rekap_hasil)}</b></td>
            <td style="text-align:center"><b>{total_km:.2f} km</b></td>
            <td style="text-align:center">—</td>
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
            title: {{ display: true, text: 'Konvergensi ACO per Klaster SPPG' }}
        }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Iterasi' }} }},
            y: {{ title: {{ display: true, text: 'Jarak Terbaik (km)' }} }}
        }}
    }}
}});
</script>
</body>
</html>"""

    with open("Grafik_Konvergensi_ACO.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("📊 Grafik konvergensi disimpan: Grafik_Konvergensi_ACO.html")


# ==========================================
# MAIN
# ==========================================
def main():
    folder_matrix = "matriks_jarak"

    try:
        df_master = pd.read_csv("sekolah_terklaster.csv")
        df_sppg   = pd.read_csv("sppg_sukolilo.csv", sep=None, engine="python")
    except FileNotFoundError as e:
        print(f"❌ File tidak ditemukan: {e}")
        return

    print("🚀 MEMULAI OPTIMASI RUTE DENGAN ANT COLONY (ACO)...\n")
    print(f"   Mode    : {'Optuna Tuning' if MODE_TUNING else 'Parameter Default'}")
    print(f"   Semut   : {N_ANTS} | Iterasi maks: {N_ITERATIONS} | Early stop: {PATIENCE}x\n")

    daftar_warna = ["blue","green","purple","orange","darkred","cadetblue","black"]
    m = folium.Map(location=[-7.2991, 112.7838], zoom_start=14)

    semua_history  = []
    nama_sppg_list = []
    rekap_hasil    = []

    json_files = sorted([f for f in os.listdir(folder_matrix) if f.endswith(".json")])

    for idx, file_json in enumerate(json_files):
        path_json = os.path.join(folder_matrix, file_json)
        with open(path_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        nama_sppg     = data["sppg_pusat"]
        matriks_jarak = data["distance_matrix_meters"]
        node_names    = data["node_names"]
        warna_rute    = daftar_warna[idx % len(daftar_warna)]

        print(f"{'='*55}")
        print(f"🐜 {nama_sppg}")
        print(f"   Node: {len(node_names)} titik (1 SPPG + {len(node_names)-1} sekolah)")

        # Tuning parameter
        if MODE_TUNING:
            opt_alpha, opt_beta, opt_decay = optuna_tuning(matriks_jarak)
        else:
            opt_alpha, opt_beta, opt_decay = DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_DECAY

        # Jalankan ACO final
        aco = AntColonyTSP(
            matriks_jarak, N_ANTS, N_ITERATIONS,
            opt_decay, opt_alpha, opt_beta, N_BEST
        )
        (best_path, best_distance), history = aco.run()

        urutan_kunjungan = [edge[0] for edge in best_path] + [best_path[-1][1]]
        jarak_km         = round(best_distance / 1000, 2)
        iterasi_berhenti = len(history)

        # FIX: print yang benar sesuai kondisi
        if iterasi_berhenti < N_ITERATIONS:
            print(f"   🛑 Early stopping di iterasi ke-{iterasi_berhenti} (konvergen)")
        else:
            print(f"   ✅ Selesai {N_ITERATIONS} iterasi penuh")
        print(f"   📏 Jarak terbaik : {jarak_km} km")

        # Susun koordinat & label rute
        df_klaster = df_master[df_master["sppg_terdekat"] == nama_sppg].reset_index(drop=True)
        baris_sppg = df_sppg[df_sppg["nama"] == nama_sppg].iloc[0]
        coord_sppg = (baris_sppg["lat"], baris_sppg["lng"])

        koordinat_rute = [coord_sppg] + list(zip(df_klaster["lat"], df_klaster["lng"]))
        info_rute      = [nama_sppg]  + df_klaster["nama_sekolah"].tolist()
        rute_nama      = [info_rute[i] for i in urutan_kunjungan]

        print(f"   🛣️  Rute: {' ➔ '.join([n for n in rute_nama])}\n")

        semua_history.append(history)
        nama_sppg_list.append(nama_sppg)
        rekap_hasil.append({
            "sppg":              nama_sppg,
            "jumlah_sekolah":    len(df_klaster),
            "jarak_rute_km":     jarak_km,
            "iterasi_berhenti":  iterasi_berhenti,
            "alpha":             round(opt_alpha, 3),
            "beta":              round(opt_beta, 3),
            "decay":             round(opt_decay, 3),
            "urutan_kunjungan":  " ➔ ".join([n for n in rute_nama]),
        })

        # Plot peta
        grup_rute = folium.FeatureGroup(
            name=f"Rute {nama_sppg} ({jarak_km} km)"
        )
        titik_urut = []

        for urutan, node_idx in enumerate(urutan_kunjungan):
            coord       = koordinat_rute[node_idx]
            nama_lokasi = info_rute[node_idx]
            titik_urut.append(coord)

            if urutan >= len(urutan_kunjungan) - 1:
                continue

            if node_idx == 0:
                folium.Marker(
                    coord,
                    popup=folium.Popup(
                        f"<b>🏭 Dapur SPPG</b><br>{nama_lokasi}<br>"
                        f"Melayani: {len(df_klaster)} sekolah<br>"
                        f"Jarak rute: {jarak_km} km<br>"
                        f"Parameter: α={opt_alpha:.2f}, β={opt_beta:.2f}, ρ={opt_decay:.2f}",
                        max_width=280
                    ),
                    tooltip=f"🏭 {nama_lokasi}",
                    icon=folium.Icon(color="red", icon="cutlery", prefix="fa"),
                ).add_to(grup_rute)
            else:
                folium.Marker(
                    coord,
                    popup=folium.Popup(
                        f"<b>🏫 {nama_lokasi}</b><br>Kunjungan ke-{urutan}",
                        max_width=200
                    ),
                    tooltip=f"Ke-{urutan}: {nama_lokasi[:30]}",
                    icon=folium.Icon(color=warna_rute, icon="book"),
                ).add_to(grup_rute)

        titik_aspal = dapatkan_jalur_aspal_osrm(titik_urut)
        folium.PolyLine(
            titik_aspal, color=warna_rute, weight=5, opacity=0.8,
            tooltip=f"{nama_sppg} — {jarak_km} km"
        ).add_to(grup_rute)
        grup_rute.add_to(m)

    # Layer control & legend
    folium.LayerControl(collapsed=False).add_to(m)
    total_km = sum(r["jarak_rute_km"] for r in rekap_hasil)
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.2);
                font-family:Arial;font-size:12px">
        <b>🐜 ACO — Distribusi MBG Sukolilo</b><br>
        <hr style="margin:5px 0">
        Total jarak    : <b>{total_km:.2f} km</b><br>
        Jumlah SPPG    : <b>{len(rekap_hasil)}</b><br>
        Jumlah sekolah : <b>{sum(r['jumlah_sekolah'] for r in rekap_hasil)}</b><br>
        Mode           : <b>{'Optuna' if MODE_TUNING else 'Default'}</b>
    </div>"""))

    # Simpan semua output
    m.save("Peta_Rute_ACO_MBG_Sukolilo.html")

    df_rekap = pd.DataFrame(rekap_hasil)
    df_rekap.to_csv("Rekap_ACO_MBG_Sukolilo.csv", index=False, encoding="utf-8-sig")

    simpan_grafik_konvergensi(semua_history, nama_sppg_list, rekap_hasil)

    print("=" * 55)
    print(f"🎉 SEMUA KLASTER BERHASIL DIPROSES!")
    print(f"🗺️  Peta        : Peta_Rute_ACO_MBG_Sukolilo.html")
    print(f"📊  Konvergensi : Grafik_Konvergensi_ACO.html")
    print(f"📄  Rekap CSV   : Rekap_ACO_MBG_Sukolilo.csv")
    print(f"📏  Total jarak : {total_km:.2f} km")


if __name__ == "__main__":
    main()