import pandas as pd
import requests
import json
import time
import os

# Fungsi pemanggil API OSRM
def get_osrm_distance_matrix(coords):
    """
    coords: list of tuples/lists e.g., [[lon1, lat1], [lon2, lat2], ...]
    OSRM membutuhkan format string: lon1,lat1;lon2,lat2;...
    """
    # Gabungkan koordinat menjadi string (OSRM butuh format Longitude, Latitude)
    coords_str = ";".join([f"{lon},{lat}" for lon, lat in coords])
    
    # Base URL OSRM Public Server (driving = mobil/motor)
    # ?annotations=distance berarti kita hanya minta matriks jarak (dalam meter)
    url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data["code"] == "Ok":
            return data["distances"] # Mengembalikan matriks (List of lists)
        else:
            print(f"  ⚠️ Peringatan: OSRM merespons dengan kode {data['code']}")
            return None
    except Exception as e:
        print(f"  ❌ Error saat menghubungi OSRM: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("Tahap 2: Generate Matriks Jarak Jalan Raya via OSRM")
    print("=" * 60)

    # 1. Load Data Klaster dan Data SPPG
    try:
        df_klaster = pd.read_csv("sekolah_terklaster.csv")
        df_sppg = pd.read_csv("sppg_sukolilo.csv", sep=None, engine='python')
    except FileNotFoundError as e:
        print(f"❌ Error: File CSV tidak ditemukan. Pastikan kamu sudah di folder yang benar.")
        exit(1)

    # Buat dictionary untuk mencari koordinat SPPG dengan cepat
    sppg_coords = {}
    for _, row in df_sppg.iterrows():
        sppg_coords[row['nama']] = (row['lng'], row['lat']) # Format: (Lon, Lat)

    # 2. Persiapkan tempat penyimpanan Matriks
    output_folder = "matriks_jarak"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Folder '{output_folder}' dibuat.")

    # 3. Proses setiap SPPG (Klaster)
    # Kita ambil daftar nama SPPG yang unik dari file sekolah_terklaster
    daftar_sppg = df_klaster['sppg_terdekat'].unique()
    
    for nama_sppg in daftar_sppg:
        print(f"\nMemproses Klaster: {nama_sppg}")
        
        # Ambil data sekolah yang HANYA di bawah SPPG ini
        sekolah_anggota = df_klaster[df_klaster['sppg_terdekat'] == nama_sppg]
        print(f" -> Terdapat {len(sekolah_anggota)} sekolah.")

        # Susun daftar koordinat untuk OSRM. 
        # ATURAN WAJIB TSP: Titik pertama (Index 0) HARUS Dapur Umum / Depot!
        lon_sppg, lat_sppg = sppg_coords[nama_sppg]
        kumpulan_koordinat = [[lon_sppg, lat_sppg]]
        
        # Buat daftar nama node (untuk dokumentasi JSON nanti)
        nama_nodes = [nama_sppg]

        for _, sekolah in sekolah_anggota.iterrows():
            kumpulan_koordinat.append([sekolah['lng'], sekolah['lat']])
            nama_nodes.append(sekolah['nama_sekolah'])

        # Tarik Matriks dari OSRM
        print(" -> Mengunduh matriks jarak dari OSRM...")
        matriks_meter = get_osrm_distance_matrix(kumpulan_koordinat)

        if matriks_meter:
            # Format file output agar aman untuk Windows/Mac (hilangkan spasi, garis miring, dll)
            safe_filename = nama_sppg.replace(" ", "_").replace("/", "").replace('"', '').replace(',', '')
            file_path = os.path.join(output_folder, f"matrix_{safe_filename}.json")
            
            # Simpan matriks dan info nama node ke dalam JSON
            data_simpan = {
                "sppg_pusat": nama_sppg,
                "jumlah_node": len(nama_nodes),
                "node_names": nama_nodes,
                "distance_matrix_meters": matriks_meter
            }
            
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(data_simpan, f, indent=4)
                
            print(f" ✅ Sukses! Disimpan ke: {file_path}")
        
        # Etika memakai public server: Beri jeda 1 detik agar tidak dianggap spam/DDoS
        time.sleep(1)

    print(f"\n{'=' * 60}")
    print("🎉 SELURUH MATRIKS JARAK BERHASIL DIBUAT!")
    print("Data siap digunakan untuk Algoritma TSP (Nearest Neighbor/GA/SA/ACO)")