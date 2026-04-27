import pandas as pd
import numpy as np
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """Menghitung jarak garis lurus (udara) antar 2 koordinat dalam Kilometer."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

if __name__ == "__main__":
    print("=" * 60)
    print("Tahap 1B: Assignment Sekolah ke SPPG (Schema Match)")
    print("=" * 60)

    # 1. Load Dataset
    try:
        df_sekolah = pd.read_csv("sekolah_sukolilo.csv")
        df_sppg = pd.read_csv("sppg_sukolilo.csv")
    except FileNotFoundError as e:
        print(f"❌ Error: File tidak ditemukan. Pastikan file CSV ada di folder yang sama.")
        exit(1)

    total_sekolah = len(df_sekolah)
    total_sppg = len(df_sppg)
    
    # LOGIKA EVEN DISTRIBUTION (6, 6, 6, 5, 5, 5, 5)
    base_capacity = total_sekolah // total_sppg
    remainder = total_sekolah % total_sppg

    sppg_max_load = {}
    for j in df_sppg.index:
        sppg_max_load[j] = base_capacity + (1 if j < remainder else 0)

    print(f"Total Sekolah : {total_sekolah}")
    print(f"Total SPPG    : {total_sppg}")
    print(f"Target Beban  : {remainder} SPPG melayani {base_capacity+1} sekolah, "
          f"{total_sppg-remainder} SPPG melayani {base_capacity} sekolah.\n")

    # 2. Hitung Semua Kombinasi Jarak (Global Greedy)
    kombinasi_jarak = []
    for i, sekolah in df_sekolah.iterrows():
        for j, sppg in df_sppg.iterrows():
            dist = haversine_distance(sekolah['lat'], sekolah['lng'], sppg['lat'], sppg['lng'])
            kombinasi_jarak.append({
                'id_sekolah': i,
                'nama_sekolah': sekolah['nama_sekolah'],
                'id_sppg': j,
                'nama_sppg': sppg['nama'], # Menggunakan kolom 'nama' sesuai request
                'alamat_sppg': sppg['alamat'],
                'jarak_km': dist
            })

    df_jarak = pd.DataFrame(kombinasi_jarak).sort_values(by='jarak_km').reset_index(drop=True)

    # 3. Proses Assignment
    sekolah_assigned = set()
    sppg_current_load = {j: 0 for j in df_sppg.index}
    hasil_assignment = []

    for _, row in df_jarak.iterrows():
        id_sek = row['id_sekolah']
        id_spg = row['id_sppg']

        if id_sek not in sekolah_assigned and sppg_current_load[id_spg] < sppg_max_load[id_spg]:
            sekolah_assigned.add(id_sek)
            sppg_current_load[id_spg] += 1
            
            hasil_assignment.append({
                'id_sekolah': id_sek,
                'nama_sekolah': row['nama_sekolah'],
                'sppg_terdekat': row['nama_sppg'],
                'alamat_sppg': row['alamat_sppg'],
                'jarak_ke_sppg_km': round(row['jarak_km'], 2)
            })

            if len(sekolah_assigned) == total_sekolah:
                break

    # 4. Gabungkan & Simpan
    df_hasil = pd.DataFrame(hasil_assignment)
    df_final = df_sekolah.merge(df_hasil[['nama_sekolah', 'sppg_terdekat', 'alamat_sppg', 'jarak_ke_sppg_km']], 
                                on='nama_sekolah', how='left')

    output_file = "sekolah_terklaster.csv"
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

    # 5. Tampilkan Rekapitulasi
    print("✅ Assignment Berhasil Diselaraskan!")
    print("\nRekapitulasi Beban Dapur Umum:")
    print("-" * 50)
    rekap = df_final.groupby(['sppg_terdekat', 'alamat_sppg'])['nama_sekolah'].count().reset_index()
    for _, row in rekap.iterrows():
        print(f"  {row['sppg_terdekat']:25s} | {row['nama_sekolah']} Sekolah | {row['alamat_sppg'][:30]}...")

    print(f"\n📄 Data tersimpan di: {output_file}")