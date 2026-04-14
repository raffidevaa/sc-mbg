""" ==============================================================
TSP MBG - Scraping Sekolah Kecamatan Sukolilo
Sumber: OpenStreetMap via Overpass API
==============================================================
"""

import requests
import pandas as pd
import time

# Multiple server mirror — kalau satu down, coba yang lain
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

QUERY_BBOX = """
[out:json][timeout:60];
(
  node["amenity"="school"](-7.318,112.765,-7.252,112.825);
  way["amenity"="school"](-7.318,112.765,-7.252,112.825);
  node["amenity"="college"](-7.318,112.765,-7.252,112.825);
  way["amenity"="college"](-7.318,112.765,-7.252,112.825);
);
out center tags;
"""

def detect_jenjang(nama, tags):
    nama_up = nama.upper()
    school_level = tags.get("school:level", "").upper()

    if any(x in nama_up for x in ["TK ", "TKN", "TKI", "PAUD", "KINDER", "RA ",
           "RAUDHATUL", "PLAYGRUP", "PLAYGROUP", " KB ", "KELOMPOK BERMAIN", "PG "]):
        return "TK/PAUD"
    elif any(x in nama_up for x in ["SMA", "SMK", "MA ", "ALIYAH", "ATAS"]) or "secondary" in school_level:
        return "SMA/SMK"
    elif any(x in nama_up for x in ["SMP", "MTS", "TSANAWIYAH", "PERTAMA"]) or "junior" in school_level:
        return "SMP"
    elif any(x in nama_up for x in ["SDN", "SD ", " SD", "SDI", "MIT ", "MI ",
             "IBTIDAIYAH", "DASAR"]) or "primary" in school_level:
        return "SD"
    else:
        return "TIDAK_DIKENAL"

def fetch_with_mirrors(query):
    """Coba semua mirror sampai ada yang berhasil."""
    for mirror in OVERPASS_MIRRORS:
        for attempt in range(2):  # 2x percobaan per mirror
            try:
                print(f"  Mencoba: {mirror.split('/')[2]} (attempt {attempt+1})...")
                resp = requests.post(
                    mirror,
                    data={"data": query},
                    timeout=60,
                    headers={"User-Agent": "TSP-MBG-Sukolilo/1.2"}
                )
                resp.raise_for_status()
                elements = resp.json().get("elements", [])
                if elements:
                    print(f"  ✅ Berhasil! {len(elements)} elemen dari {mirror.split('/')[2]}")
                    return elements
                else:
                    print(f"  ⚠️  Response kosong, coba mirror berikutnya...")
                    break
            except requests.exceptions.Timeout:
                print(f"  ⏱️  Timeout, tunggu 5 detik...")
                time.sleep(5)
            except Exception as e:
                print(f"  ❌ Error: {e}")
                time.sleep(3)
                break
        time.sleep(2)
    return []

def parse_elements(elements):
    schools = []
    for el in elements:
        tags = el.get("tags", {})

        if el["type"] == "node":
            lat, lng = el.get("lat"), el.get("lon")
        else:
            center = el.get("center", {})
            lat, lng = center.get("lat"), center.get("lon")

        if not lat or not lng:
            continue

        nama = (
            tags.get("name") or
            tags.get("name:id") or
            tags.get("official_name") or ""
        )
        if not nama:
            continue

        jenjang = detect_jenjang(nama, tags)
        if jenjang not in ["SD", "SMP", "SMA/SMK"]:
            continue

        schools.append({
            "osm_id":       el.get("id"),
            "nama_sekolah": nama.strip().title(),
            "jenjang":      jenjang,
            "alamat":       tags.get("addr:full", tags.get("addr:street", "")),
            "kelurahan":    tags.get("addr:suburb", tags.get("addr:village",
                            tags.get("addr:quarter", ""))),
            "kecamatan":    "Sukolilo",
            "kota":         "Surabaya",
            "lat":          round(lat, 7),
            "lng":          round(lng, 7),
        })
    return schools

if __name__ == "__main__":
    print("=" * 60)
    print("TSP MBG - Scraping Sekolah Kecamatan Sukolilo")
    print("Mode: Multi-mirror + retry")
    print("=" * 60)

    print("\nMengambil data dari OSM...")
    elements = fetch_with_mirrors(QUERY_BBOX)

    if not elements:
        print("\n❌ Semua mirror gagal. Kemungkinan:")
        print("   1. Koneksi internet bermasalah")
        print("   2. Semua server Overpass sedang overload")
        print("   Coba lagi beberapa menit kemudian.")
        exit(1)

    schools = parse_elements(elements)
    if not schools:
        print("❌ Tidak ada sekolah SD/SMP/SMA ditemukan.")
        exit(1)

    df = pd.DataFrame(schools)
    before = len(df)
    df = df.drop_duplicates(subset=["lat", "lng"])
    df = df.drop_duplicates(subset=["nama_sekolah"])
    df = df.sort_values(["jenjang", "nama_sekolah"]).reset_index(drop=True)
    df.index += 1

    df.to_csv("sekolah_sukolilo_.csv", index_label="no", encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"✅ Total sekolah (SD/SMP/SMA): {len(df)}")
    print(f"   Duplikat dihapus: {before - len(df)}")
    print(f"\nRekap per jenjang:")
    for jenjang, count in df["jenjang"].value_counts().items():
        print(f"  {jenjang:10s}: {count} sekolah")

    print(f"\nDaftar lengkap:")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"  [{row.name:2d}] {row['jenjang']:8s} | {row['nama_sekolah'][:40]:40s} | {row['kelurahan']}")

    print(f"\n📄 Disimpan: sekolah_sukolilo.csv")
