import pandas as pd
import numpy as np
import json
import folium
import os
import requests

# ==========================================
# KONFIGURASI PARAMETER ANT COLONY (ACO)
# ==========================================
N_ANTS       = 5 #jumlah semut per iterasi
N_ITERATIONS = 100 #jumlah iterasi
DECAY        = 0.3 #laju evaporasi pheromone (rho) (0.3 = 30% per iterasi)
ALPHA        = 0.5 #pengaruh pheromone - seberapa kuat semut mengikuti jejak (0 = tidak ada pengaruh, >1 = lebih kuat) 
BETA         = 1 #pengaruh heuristik (jarak) - seberapa kuat semut memilih rute lebih pendek (0 = tidak ada pengaruh, >1 = lebih kuat)
N_BEST       = 1 #hanya N semut terbaik yang deposit pheromone (0 = semua semut deposit, >0 = hanya N terbaik)   


# ==========================================
# CLASS ANT COLONY TSP
# ==========================================
class AntColonyTSP:
    def __init__(self, distance_matrix, n_ants, n_iterations, decay, alpha, beta, n_best=5):
        self.distances   = np.array(distance_matrix, dtype=float)
        np.fill_diagonal(self.distances, np.inf)

        self.pheromone   = np.ones(self.distances.shape) / len(distance_matrix)
        self.all_inds    = range(len(distance_matrix))
        self.n_ants      = n_ants
        self.n_iterations= n_iterations
        self.decay       = decay
        self.alpha       = alpha
        self.beta        = beta
        self.n_best      = n_best

    def run(self):
        all_time_shortest_path = (None, np.inf)
        history = []  #simpan konvergensi tiap iterasi

        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()

            #evaporasi dan lanjut ke deposit
            self.pheromone *= (1 - self.decay)
            #DECAY = 0.3 berarti pheromone berkurang 30% setiap iterasi, sehingga terpaksa eksplor rute baru

            #urutkan semua rute, ambil N_BEST=1 terbaik saja
            sorted_paths = sorted(all_paths, key=lambda x: x[1])
            self.spread_pheromone(sorted_paths[:self.n_best])

            # Update global best
            shortest_this_iter = sorted_paths[0]
            if shortest_this_iter[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_this_iter

            history.append(all_time_shortest_path[1])

        return all_time_shortest_path, history

    def gen_all_paths(self):
        return [
            (path := self.gen_path(0), self.path_dist(path))
            for _ in range(self.n_ants)
        ]

    def gen_path(self, start):
        path    = []
        visited = {start} #sppg sebagai depot selalu dikunjungi pertama
        prev    = start #posisi semut mulai dari depot

        for _ in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move)) #mencacat edge (u,v) untuk perhitungan jarak
            prev = move
            visited.add(move)

        path.append((prev, start))  # kembali ke depot (menuutup loop)
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone         = np.copy(pheromone)
        pheromone[list(visited)] = 0 # kota yang sudah dikunjungi = probabilitas 0

        with np.errstate(divide="ignore", invalid="ignore"):
            heuristic = np.where(dist > 0, 1.0 / dist, 0.0)  # heuristic = 1/jarak → makin dekat, makin menarik

        row = (pheromone ** self.alpha) * (heuristic ** self.beta)
        # rumus transisi ACO:
        # probabilitas ∝ τ^α × η^β
        # τ = pheromone, η = heuristic (1/jarak)
        # α=0.5 → pheromone tidak terlalu dominan
        # β=1   → heuristic tidak terlalu dominan
        # sehingga semut lebih eksploratif

        total = row.sum()
        if total == 0:
            # hindari division by zero - terjadi kalau semua kolom yang belum dikunjungi punya prob=0
            unvisited = [j for j in self.all_inds if j not in visited]
            return np.random.choice(unvisited) #pilih random

        norm_row = row / total #normalisasi jadi distribusi probabilitas
        return np.random.choice(list(self.all_inds), 1, p=norm_row)[0]
    
    def spread_pheromone(self, best_paths):
        # best_paths = hanya N_BEST=1 rute terpendek
        for path, dist in best_paths:
            for move in path:
                self.pheromone[move] += 1.0 / dist
                # deposit ∝ 1/jarak
                # rute lebih pendek → 1/dist lebih besar → jejak lebih kuat
                # semut berikutnya lebih tertarik jalur ini

    def path_dist(self, path):
        return sum(self.distances[u][v] for u, v in path)


# ==========================================
# FUNGSI TARIK GEOMETRI JALAN ASPAL (OSRM)
# ==========================================
def dapatkan_jalur_aspal_osrm(koordinat_rute_urut):
    """
    Ambil geometri jalan nyata via OSRM.
    Fallback: kembalikan koordinat lurus jika OSRM gagal.
    """
    #ada 3 error handling: timeout, HTTP error, dan error lain (misal JSON decode error)
    #fungsi nya hanya untuk visualiasasi peta dan tidak memengaruhi hasil optimasi ACO
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in koordinat_rute_urut])
    url = (
        f"http://router.project-osrm.org/route/v1/driving/{coords_str}"
        f"?overview=full&geometries=geojson"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == "Ok":
            jalur_asli = data["routes"][0]["geometry"]["coordinates"]
            print(f"Rute aspal berhasil diambil dari OSRM")
            return [(lat, lon) for lon, lat in jalur_asli]
        else:
            print(f"   ⚠️ OSRM code bukan Ok: {data.get('code')} → pakai garis lurus")
    except requests.exceptions.Timeout:
        print(f"   ⚠️ OSRM timeout → pakai garis lurus")
    except Exception as e:
        print(f"   ⚠️ OSRM gagal: {e} → pakai garis lurus")

    return koordinat_rute_urut  # fallback


# ==========================================
# FUNGSI PLOT GRAFIK KONVERGENSI
# ==========================================
def simpan_grafik_konvergensi(semua_history, nama_sppg_list):
    """
    Simpan grafik konvergensi ACO per SPPG sebagai HTML interaktif
    menggunakan chart.js (tanpa matplotlib).
    """
    labels_js   = json.dumps(list(range(1, N_ITERATIONS + 1)))
    warna_list  = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6","#1ABC9C","#E67E22"]

    datasets = []
    for i, (history, nama) in enumerate(zip(semua_history, nama_sppg_list)):
        history_km = [round(d / 1000, 3) for d in history]
        warna = warna_list[i % len(warna_list)]
        datasets.append(f"""{{
            label: "{nama[-30:]}",
            data: {json.dumps(history_km)},
            borderColor: "{warna}",
            backgroundColor: "{warna}22",
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
            tension: 0.3
        }}""")

    datasets_js = ",\n".join(datasets)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Konvergensi ACO - MBG Sukolilo</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }}
        h2   {{ color: #333; }}
        .container {{ background: white; padding: 20px; border-radius: 8px;
                      box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 900px; margin: auto; }}
    </style>
</head>
<body>
<div class="container">
    <h2>🐜 Konvergensi ACO — Distribusi MBG Kecamatan Sukolilo</h2>
    <p>Sumbu X: Iterasi | Sumbu Y: Jarak terbaik (km)</p>
    <canvas id="chart" height="400"></canvas>
</div>
<script>
const ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {{
    type: 'line',
    data: {{
        labels: {labels_js},
        datasets: [
            {datasets_js}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{ position: 'bottom' }},
            title: {{
                display: true,
                text: 'Konvergensi ACO per Klaster SPPG'
            }}
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
# Main function: proses semua klaster SPPG, optimasi ACO, dan visualisasi peta
# ==========================================
def main():
    #load data master dan sppg
    folder_matrix = "matriks_jarak"

    try:
        df_master = pd.read_csv("sekolah_terklaster.csv")
        df_sppg   = pd.read_csv("sppg_sukolilo.csv", sep=None, engine="python")
    except FileNotFoundError as e:
        print(f"❌ File tidak ditemukan: {e}")
        return

    print("MEMULAI OPTIMASI RUTE DENGAN ANT COLONY (ACO)...\n")

    daftar_warna = ["blue","green","purple","orange","darkred","cadetblue","black"]
    m = folium.Map(location=[-7.2991, 112.7838], zoom_start=14)

    semua_history    = []
    nama_sppg_list   = []
    rekap_hasil      = []

    json_files = sorted([f for f in os.listdir(folder_matrix) if f.endswith(".json")])

    for idx, file_json in enumerate(json_files):
        path_json = os.path.join(folder_matrix, file_json)
        with open(path_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        nama_sppg     = data["sppg_pusat"]
        matriks_jarak = data["distance_matrix_meters"]
        #node_names sekarang dipakai untuk validasi
        node_names    = data["node_names"]
        warna_rute    = daftar_warna[idx % len(daftar_warna)]

        print(f"🐜 Memproses: {nama_sppg[-35:]}")
        print(f"   Node: {len(node_names)} titik (1 SPPG + {len(node_names)-1} sekolah)")

        #menjalankan ACO untuk dapat best_path dan best_distance
        aco = AntColonyTSP(
            matriks_jarak, N_ANTS, N_ITERATIONS,
            DECAY, ALPHA, BETA, N_BEST
        )
        
        (best_path, best_distance), history = aco.run()  # terima history

        #menyusun urutkan kunjungan dari best_path
        urutan_kunjungan = [edge[0] for edge in best_path] + [best_path[-1][1]]
        jarak_km         = round(best_distance / 1000, 2)

        # Susun koordinat dan label rute
        df_klaster  = df_master[df_master["sppg_terdekat"] == nama_sppg].reset_index(drop=True)
        baris_sppg  = df_sppg[df_sppg["nama"] == nama_sppg].iloc[0]
        coord_sppg  = (baris_sppg["lat"], baris_sppg["lng"])

        koordinat_rute = [coord_sppg] + list(zip(df_klaster["lat"], df_klaster["lng"]))
        info_rute      = [nama_sppg]  + df_klaster["nama_sekolah"].tolist()

        rute_nama = [info_rute[i] for i in urutan_kunjungan]
        print(f"   ✅ Jarak Terbaik : {jarak_km} km")
        print(f"   🛣️  Rute: {' ➔ '.join([n[:20] for n in rute_nama])}\n")

        # Simpan untukrekap & konvergensi
        semua_history.append(history)
        nama_sppg_list.append(nama_sppg)
        rekap_hasil.append({
            "sppg":             nama_sppg,
            "jumlah_sekolah":   len(df_klaster),
            "jarak_rute_km":    jarak_km,
            "urutan_kunjungan": " ➔ ".join([n[:30] for n in rute_nama]),
        })

        # Membuat layer peta
        grup_rute = folium.FeatureGroup(
            name=f"Rute {nama_sppg[-25:]} ({jarak_km} km)"
        )

        titik_urut = []
        for urutan, node_idx in enumerate(urutan_kunjungan):
            coord        = koordinat_rute[node_idx]
            nama_lokasi  = info_rute[node_idx]
            titik_urut.append(coord)

            # Jangan buat marker untuk titik terakhir (duplikat depot)
            if urutan >= len(urutan_kunjungan) - 1:
                continue

            if node_idx == 0:
                folium.Marker(
                    coord,
                    popup=folium.Popup(
                        f"<b>🏭 Dapur SPPG</b><br>{nama_lokasi}<br>"
                        f"Melayani: {len(df_klaster)} sekolah<br>"
                        f"Total rute: {jarak_km} km",
                        max_width=250
                    ),
                    tooltip=folium.Tooltip(
                        f"🏭 {nama_lokasi[-25:]}", permanent=False
                    ),
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

        # Ambil geometri jalan aspal via OSRM
        titik_aspal = dapatkan_jalur_aspal_osrm(titik_urut)

        folium.PolyLine(
            titik_aspal,
            color=warna_rute,
            weight=5,
            opacity=0.8,
            tooltip=f"{nama_sppg[-25:]} — {jarak_km} km",
        ).add_to(grup_rute)

        grup_rute.add_to(m)

    # ── Layer control & simpan peta ──
    folium.LayerControl(collapsed=False).add_to(m)

    # Legend total
    total_km    = sum(r["jarak_rute_km"] for r in rekap_hasil)
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.2);
                font-family:Arial;font-size:12px">
        <b>🐜 ACO — Distribusi MBG Sukolilo</b><br>
        <hr style="margin:5px 0">
        Total jarak : <b>{total_km:.2f} km</b><br>
        Jumlah SPPG : <b>{len(rekap_hasil)}</b><br>
        Jumlah sekolah : <b>{sum(r['jumlah_sekolah'] for r in rekap_hasil)}</b>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    nama_peta = "Peta_Rute_ACO_MBG_Sukolilo.html"
    m.save(nama_peta)

    # ── Simpan rekap CSV ──
    pd.DataFrame(rekap_hasil).to_csv(
        "Rekap_ACO_MBG_Sukolilo.csv", index=False, encoding="utf-8-sig"
    )

    # ── Grafik konvergensi ──
    simpan_grafik_konvergensi(semua_history, nama_sppg_list)

    print("=" * 60)
    print(f"🎉 SEMUA KLASTER BERHASIL DIPROSES!")
    print(f"🗺️  Peta        : {nama_peta}")
    print(f"📊  Konvergensi : Grafik_Konvergensi_ACO.html")
    print(f"📄  Rekap CSV   : Rekap_ACO_MBG_Sukolilo.csv")
    print(f"📏  Total jarak : {total_km:.2f} km")


if __name__ == "__main__":
    main()