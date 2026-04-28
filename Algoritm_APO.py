import pandas as pd
import numpy as np
import json
import folium
import os
import random
import requests

# ==========================================
# KONFIGURASI PARAMETER ANT COLONY (ACO)
# ==========================================
N_ANTS = 20
N_ITERATIONS = 50
DECAY = 0.1
ALPHA = 1
BETA = 2

class AntColonyTSP:
    def __init__(self, distance_matrix, n_ants, n_iterations, decay, alpha, beta):
        self.distances = np.array(distance_matrix)
        np.fill_diagonal(self.distances, np.inf) 
        self.pheromone = np.ones(self.distances.shape) / len(distance_matrix)
        self.all_inds = range(len(distance_matrix))
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)

        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.shortest_path(all_paths)[1])
            shortest_path = self.shortest_path(all_paths)
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone = self.pheromone * (1 - self.decay)
            
        return all_time_shortest_path

    def spread_pheromone(self, all_paths, shortest_path_length):
        for path, dist in all_paths:
            for move in path:
                self.pheromone[move] += 1.0 / dist 

    def shortest_path(self, all_paths):
        return min(all_paths, key=lambda x: x[1])

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
            
        path.append((prev, start))
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0 
        
        row = (pheromone ** self.alpha) * ((1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move

    def path_dist(self, path):
        total_dist = 0
        for (u, v) in path:
            total_dist += self.distances[u][v]
        return total_dist

# ==========================================
# FUNGSI TARIK GEOMETRI JALAN ASPAL (OSRM)
# ==========================================
def dapatkan_jalur_aspal_osrm(koordinat_rute_urut):
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in koordinat_rute_urut])
    url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get("code") == "Ok":
            jalur_asli = data["routes"][0]["geometry"]["coordinates"]
            return [(lat, lon) for lon, lat in jalur_asli]
    except Exception as e:
        print(f"   ⚠️ Gagal menarik rute aspal: {e}")
    
    return koordinat_rute_urut

# ==========================================
# FUNGSI UTAMA 
# ==========================================
def main():
    folder_matrix = "matriks_jarak"
    
    try:
        df_master = pd.read_csv("sekolah_terklaster.csv")
        df_sppg = pd.read_csv("sppg_sukolilo.csv", sep=None, engine='python')
    except FileNotFoundError:
        print("❌ File CSV tidak ditemukan! Pastikan skrip ini ada di folder sc-mbg-main.")
        return

    print("🚀 MEMULAI OPTIMASI RUTE DENGAN ANT COLONY (ACO)...\n")

    daftar_warna = ['blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'black']
    m = folium.Map(location=[-7.2891, 112.7838], zoom_start=13,)

    for idx, file_json in enumerate(os.listdir(folder_matrix)):
        if not file_json.endswith(".json"): continue
        
        path_json = os.path.join(folder_matrix, file_json)
        with open(path_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        nama_sppg = data['sppg_pusat']
        matriks_jarak = data['distance_matrix_meters']
        node_names = data['node_names']
        warna_rute = daftar_warna[idx % len(daftar_warna)]
        
        print(f"🐜 Memproses Klaster: {nama_sppg}")
        
        aco = AntColonyTSP(matriks_jarak, N_ANTS, N_ITERATIONS, DECAY, ALPHA, BETA)
        best_path, best_distance = aco.run()
        
        urutan_kunjungan = [edge[0] for edge in best_path] + [best_path[-1][1]]
        jarak_km = round(best_distance / 1000, 2)

        df_klaster = df_master[df_master['sppg_terdekat'] == nama_sppg].reset_index()
        baris_sppg = df_sppg[df_sppg['nama'] == nama_sppg].iloc[0]
        coord_sppg = (baris_sppg['lat'], baris_sppg['lng'])
        
        koordinat_rute = [coord_sppg]
        info_rute = [nama_sppg] 
        
        for _, row in df_klaster.iterrows():
            koordinat_rute.append((row['lat'], row['lng']))
            info_rute.append(f"{row['nama_sekolah']}")

        rute_nama_sekolah = [info_rute[i] for i in urutan_kunjungan]
        print(f"   ✅ Jarak Terbaik: {jarak_km} KM")
        print(f"   🛣️ Rute: {' ➔ '.join(rute_nama_sekolah)}\n")

        grup_rute = folium.FeatureGroup(name=f"Rute {nama_sppg}")
        
        titik_garis_lurus = []
        for urutan, node_idx in enumerate(urutan_kunjungan):
            coord = koordinat_rute[node_idx]
            nama_lokasi = info_rute[node_idx]
            titik_garis_lurus.append(coord)
            
            if urutan < len(urutan_kunjungan) - 1: 
                if node_idx == 0:
                    # Ikon Dapur (cutlery) untuk SPPG
                    folium.Marker(
                        coord, 
                        popup=f"<b>Dapur: {nama_lokasi}</b>", 
                        tooltip=folium.Tooltip(f"Dapur: {nama_lokasi}", permanent=True, direction='top', className='sppg-label'),
                        icon=folium.Icon(color='red', icon='cutlery', prefix='fa')
                    ).add_to(grup_rute)
                else:
                    #Ikon Sekolah ('university') untuk sekolah
                    folium.Marker(
                        coord, 
                        popup=f"<b>{nama_lokasi}</b><br>Kunjungan ke-{urutan}", 
                        tooltip=f"Ke-{urutan}: {nama_lokasi}",
                        icon=folium.Icon(color=warna_rute, icon='book')
                    ).add_to(grup_rute)

        titik_garis_aspal = dapatkan_jalur_aspal_osrm(titik_garis_lurus)

        folium.PolyLine(
            titik_garis_aspal, 
            color=warna_rute, 
            weight=5, 
            opacity=0.8, 
            tooltip=f"Jalur {nama_sppg} ({jarak_km} KM)"
        ).add_to(grup_rute)
        
        grup_rute.add_to(m)

    folium.LayerControl().add_to(m)

    nama_file_output = "Peta_Rute_ACO_MBG_Sukolilo.html"
    m.save(nama_file_output)
    print("=" * 60)
    print(f"🎉 SEMUA KLASTER BERHASIL DIPROSES!")
    print(f"🗺️ Peta Gabungan tersimpan di: {nama_file_output}")

if __name__ == "__main__":
    main()