import pandas as pd
import numpy as np
import json
import folium
import os
import random
import math
import requests

# ==========================================
# PARAMETER SA
# ==========================================
T_INIT = 1000
ALPHA = 0.995
T_MIN = 1

# ==========================================
# CLASS SIMULATED ANNEALING
# ==========================================
class SimulatedAnnealingTSP:
    def __init__(self, distance_matrix):
        self.distances = np.array(distance_matrix)
        self.n = len(distance_matrix)

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

        T = T_INIT

        while T > T_MIN:
            neighbor = self.get_neighbor(current)

            delta = self.fitness(neighbor) - self.fitness(current)

            if delta < 0:
                current = neighbor
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current = neighbor

            if self.fitness(current) < self.fitness(best):
                best = current

            T *= ALPHA

        return best, self.fitness(best)

# ==========================================
# OSRM ROUTE
# ==========================================
def dapatkan_jalur_aspal_osrm(koordinat_rute_urut):
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in koordinat_rute_urut])
    url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get("code") == "Ok":
            jalur = data["routes"][0]["geometry"]["coordinates"]
            return [(lat, lon) for lon, lat in jalur]
    except:
        pass
    
    return koordinat_rute_urut

# ==========================================
# MAIN
# ==========================================
def main():
    folder_matrix = "matriks_jarak"

    df_master = pd.read_csv("sekolah_terklaster.csv")
    df_sppg = pd.read_csv("sppg_sukolilo.csv", sep=None, engine='python')

    print("Simulated Annealing for Route Optimization\n")

    m = folium.Map(location=[-7.2891, 112.7838], zoom_start=13)
    warna_list = ['blue', 'green', 'purple', 'orange', 'red', 'black']

    for idx, file_json in enumerate(os.listdir(folder_matrix)):
        if not file_json.endswith(".json"): continue

        with open(os.path.join(folder_matrix, file_json), 'r') as f:
            data = json.load(f)

        nama_sppg = data['sppg_pusat']
        matrix = data['distance_matrix_meters']
        node_names = data['node_names']

        print(f"Processing cluster: {nama_sppg}")

        sa = SimulatedAnnealingTSP(matrix)
        best_route, best_distance = sa.run()

        jarak_km = round(best_distance / 1000, 2)
        print(f"Total Distance : {jarak_km} km")

        df_klaster = df_master[df_master['sppg_terdekat'] == nama_sppg].reset_index()
        baris_sppg = df_sppg[df_sppg['nama'] == nama_sppg].iloc[0]

        koordinat = [(baris_sppg['lat'], baris_sppg['lng'])]
        nama_lokasi = [nama_sppg]

        for _, row in df_klaster.iterrows():
            koordinat.append((row['lat'], row['lng']))
            nama_lokasi.append(row['nama_sekolah'])

        rute_nama = [nama_lokasi[i] for i in best_route]

        print("Route Sequence :")
        for i, lokasi in enumerate(rute_nama):
            print(f"      {i+1}. {lokasi}")
            print()

        group = folium.FeatureGroup(name=nama_sppg)
        warna = warna_list[idx % len(warna_list)]

        titik = [koordinat[i] for i in best_route]
        titik.append(titik[0])

        jalur = dapatkan_jalur_aspal_osrm(titik)

        for i, idx_node in enumerate(best_route):
            coord = koordinat[idx_node]
            nama = nama_lokasi[idx_node]

            if idx_node == 0:
                folium.Marker(
                    coord,
                    popup=f"Depot: {nama}",
                    icon=folium.Icon(color='red')
                ).add_to(group)
            else:
                folium.Marker(
                    coord,
                    popup=f"{i}. {nama}",
                    icon=folium.Icon(color=warna)
                ).add_to(group)

        folium.PolyLine(jalur, color=warna, weight=5).add_to(group)
        group.add_to(m)

    folium.LayerControl().add_to(m)
    m.save("Peta_Rute_SA_MBG_Sukolilo.html")

    print("\nProcess completed. Map has been saved as 'Peta_Rute_SA_MBG_Sukolilo.html'.")

if __name__ == "__main__":
    main()