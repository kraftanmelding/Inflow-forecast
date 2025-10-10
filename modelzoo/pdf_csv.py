# pdf_to_csv.py
import os
import re
from glob import glob
import pandas as pd
import PyPDF2 as pypdf
# from pyproj import Transformer  # <- not used; remove if unnecessary

# --- CONFIG (defaults) ---
BASE_DIR = "..."
PATTERN_PREFIX = "Nedbørfeltparam-"
OUTPUT_CSV = os.path.join("/home/ubuntu/myproject/operational/", "feltparam_sk.csv")

# --- HELPERS ---
def read_pdf_text(pdf_path: str) -> str:
    txt_chunks = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text() or ""
            txt_chunks.append(t)
    return "\n".join(txt_chunks)

_key_val_colon = re.compile(r"^(?P<key>[^:]+):\s*(?P<val>.+)$")
_key_val_trailing = re.compile(
    r"^(?P<key>.*?)(?P<val>-?\d[\d\.,]*)\s*(?P<unit>[%°A-Za-zµ/²³\-\*/\.\^·×\(\)/–— ]*)$"
)

def parse_lines_to_pairs(text: str) -> dict:
    pairs = {}
    for raw in text.splitlines():
        line = " ".join(raw.split()).strip()
        if not line:
            continue
        m = _key_val_colon.match(line)
        if m:
            key = m.group("key").strip().rstrip(".")
            val = m.group("val").strip()
        else:
            m2 = _key_val_trailing.match(line)
            if not m2:
                continue
            key = m2.group("key").strip().rstrip(".")
            val = m2.group("val").strip()
        key = key.replace("  ", " ")
        try:
            val_num = float(val.replace(" ", "").replace(",", "."))
            pairs[key] = val_num
        except Exception:
            pairs[key] = val
    return pairs

def catchment_name_from_file(pdf_path: str, pattern_prefix: str | None = PATTERN_PREFIX) -> str:
    name = os.path.splitext(os.path.basename(pdf_path))[0]
    if pattern_prefix and name.startswith(pattern_prefix):
        return name[len(pattern_prefix):].lower()
    return name.lower()

# --- CORE: build a DF from an explicit list of PDF paths ---
def make_df(pdf_files: list[str], pattern_prefix: str = PATTERN_PREFIX) -> pd.DataFrame:
    """Return a DataFrame built from the given list of PDF files."""
    if not pdf_files:
        raise FileNotFoundError("No PDF files were provided to make_df(pdf_files=...).")

    data_by_catchment = {}
    for pdf in sorted(pdf_files):
        catchment = catchment_name_from_file(pdf, pattern_prefix)
        text = read_pdf_text(pdf)
        pairs = parse_lines_to_pairs(text)
        if not pairs:
            print(f"[WARN] No key/value pairs parsed from: {os.path.basename(pdf)}")
        data_by_catchment[catchment] = pairs

    df = pd.DataFrame.from_dict(data_by_catchment, orient="index")
    df.index.name = "basin_id"

    nums = df["Beregn.punkt"].str.findall(r"\d+")
    df["utm_east_z33"]  = pd.to_numeric(nums.str[0], errors="coerce")
    df["utm_north_z33"] = pd.to_numeric(nums.str[1], errors="coerce")

    cols_to_remove = [
        'Kartdatum','Projeksjon','Vassdragsnr.','Rapportdato','Kommune.','Fylke.','Beregn.punkt',
        'Vassdrag.','Norges vassdrags- og energidirektoratKartbakgrunn','Dreneringstetthet (DT)0.3km',
       'Dreneringstetthet (DT)0.5km', 'Dreneringstetthet (DT)0.6km',
       'Dreneringstetthet (DT)0.7km', 'Dreneringstetthet (DT)0.9km',
       'Dreneringstetthet (DT)1km', 'Dreneringstetthet (DT)2.9km',
       'Dreneringstetthet (DT)3.2km', 'Dreneringstetthet (DT)3.4km',
       'Dreneringstetthet (DT)3.9km', 'Dreneringstetthet (DT)4.5km',
       'Dreneringstetthet (DT)5.4km', 'Dreneringstetthet (DT)5.6km',
       'Dreneringstetthet (DT)8.9km','Dreneringstetthet (DT)0.8km',
        'Dreneringstetthet (DT)1.1km','Dreneringstetthet (DT)1.2km','Dreneringstetthet (DT)1.3km',
        'Dreneringstetthet (DT)1.4km','Dreneringstetthet (DT)1.5km','Dreneringstetthet (DT)1.6km',
        'Dreneringstetthet (DT)1.7km','Dreneringstetthet (DT)1.8km','Dreneringstetthet (DT)1.9km',
        'Dreneringstetthet (DT)2.1km','Dreneringstetthet (DT)2.2km','Dreneringstetthet (DT)2.3km',
        'Dreneringstetthet (DT)2.4km','Dreneringstetthet (DT)2.5km','Dreneringstetthet (DT)2.6km',
        'Dreneringstetthet (DT)2.7km','Dreneringstetthet (DT)2.8km','Dreneringstetthet (DT)2km',
        'Dreneringstetthet (DT)3.1km','Dreneringstetthet (DT)3.3km','Dreneringstetthet (DT)3.5km',
        'Dreneringstetthet (DT)3.6km','Dreneringstetthet (DT)3km','Dreneringstetthet (DT)4.1km',
        '(1991','Effektiv sjø - Tilløp (ASE-T)','Elvegradent1085 (EG,1085)','Elvegradient (EG)',
        'Feltlengde (FL)','Feltlengde - Tilløp (FL-T)','Fylke','GUID','Høyde','Kommune',
        'Uklassifisert areal (AREST)','Usikkerhet middelavrenning','Vassdrag','Vassdragsnr',
        'Effektiv sjø – Tilløp (ASE-T)','Feltlengde – Tilløp (FL-T)'
    ]
    df = df.drop(columns=cols_to_remove, errors='ignore')
    

    rename_map = {
        'Areal (A)': 'area_total',
        'Årlig middelavrenning (QN)': 'specific_runoff',
        'Arlig middelavrenning': 'annual_runoff',
        'Bre (ABRE)': 'perc_glacier',
        'Dyrket mark (AJORD)': 'perc_agricul',
        'Elvleengde (EL)': 'length_km_river',
        'Helning': 'slope',
        'Høyde10': 'height_hypso_10',
        'Høyde20': 'height_hypso_20',
        'Høyde30': 'height_hypso_30',
        'Høyde40': 'height_hypso_40',
        'Høyde50': 'height_hypso_50',
        'Høyde60': 'height_hypso_60',
        'Høyde70': 'height_hypso_70',
        'Høyde80': 'height_hypso_80',
        'Høyde90': 'height_hypso_90',
        'HøydeMAX': 'height_maximum',
        'HøydeMIN': 'height_minimum',
        'Leire (ALEIRE)': 'perc_clay',
        'Myr (AMYR)': 'perc_bog',
        'Nedbør desember - februar': 'mean_winter_prec',
        'Nedbør juni - august': 'mean_summer_prec',
        'Sjø (ASJO)': 'perc_lake',
        'Effektiv sjø (ASE)': 'perc_eff_lake',
        'Skog (ASKOG)': 'perc_forest',
        'Snaufjell (ASF)': 'perc_mountain',
        'Sommertemperatur': 'mean_summer_temp',
        'Urban (AU)': 'perc_urban',
        'Vintertemperatur': 'mean_winter_temp',
        'Årstemperatur': 'mean_annual_temp',
        'lon': 'longitude',
        'lat': 'latitude',
    }
    df = df.rename(columns=rename_map)
    df = df.reset_index()
    other_cols = sorted([c for c in df.columns if c != "basin_id"])
    df = df[["basin_id"] + other_cols]
    return df





