# PRK POI COUNT (from PRK pickles)
# PRK POI COUNT (from PRK pickles)
# PRK POI COUNT (from PRK pickles)

import re, json, pickle
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from pathlib import Path
from collections import OrderedDict

CSV_PATH     = "/home/donggyu/donggyu/RegionalEstimationLLM/*****temporary/NK DB DATA SET - TN_POI_북한.csv"                
PRK_GEOM_PKL = "aoi_2024_geom.pickle"         
PRK_LOC_PKL  = "aoi_2024_loc_fixed.pickle"          
OUT_JSON     = "PRK2010_2024_poi.json"       
ADM0_NAME    = "PRK2010_2024"                     
SOURCE_KEEP  = "지방자치단체인허가정보"    

# CSV header map
COLMAP = {
    "고유식별자 아이디": "poi_id",
    "관심지점 명칭": "poi_name",
    "관심지점 분류 설명": "class_desc",
    "자료 출처": "source",
    "도로명 주소 명칭": "road_addr",
    "지번 주소 명칭": "jibun_addr",
    "자료수집 일시": "collected_at",
    "객체변동 일시": "changed_at",
    "x(EPSG:5179)": "x_5179",
    "y(EPSG:5179)": "y_5179",
}

DESC_TEXT_NAMES = "List of production facilities"
TYPE_TAG        = "default"  

PREFERRED_FIELD_ORDER = [
    "ADM0", "ADM1", "ADM2",
    "no2", "so2", "co", "lst",
    "production_facilities",        
    "repr_loc", "weight",
]

def wrap_names(names):
    # Ensure plain strings
    clean = [str(x) for x in names if isinstance(x, (str, int, float))]
    return {"val": clean, "desc": DESC_TEXT_NAMES, "type": TYPE_TAG, "weight": None}

def order_block(block: dict) -> OrderedDict:
    od = OrderedDict()
    for k in PREFERRED_FIELD_ORDER:
        if k in block:
            od[k] = block[k]
    for k in block:
        if k not in od:
            od[k] = block[k]
    return od

def load_prk_aoi(prk_geom_pkl: str, prk_loc_pkl: str) -> gpd.GeoDataFrame:
    with open(prk_geom_pkl, "rb") as f:
        geom_dict = pickle.load(f)
    with open(prk_loc_pkl, "rb") as f:
        loc_dict = pickle.load(f)

    rows = []
    for code, geom in geom_dict.items():
        if not isinstance(geom, (Polygon, MultiPolygon)):
            continue
        meta = loc_dict.get(code, {})
        rows.append({
            "gid": code,
            "ADM0": ADM0_NAME,
            "ADM1": meta.get("ADM1", ""),
            "ADM2": meta.get("ADM2", ""),
            "geometry": geom
        })

    if not rows:
        raise ValueError("No valid (Multi)Polygons found in PRK_geom.pickle")

    aoi = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if aoi.crs is None:
        aoi.set_crs("EPSG:4326", inplace=True)
    elif aoi.crs.to_string().upper() != "EPSG:4326":
        aoi = aoi.to_crs("EPSG:4326")
    return aoi[["gid", "ADM0", "ADM1", "ADM2", "geometry"]]

def normalize_name(s: str) -> str:
    if s is None:
        return ""
    # strip, collapse whitespace, remove zero-width spaces
    s = re.sub(r"[\u200b-\u200d\uFEFF]", "", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main(csv_path=CSV_PATH, prk_geom_pkl=PRK_GEOM_PKL, prk_loc_pkl=PRK_LOC_PKL, out_json=OUT_JSON):
    aoi_idx = load_prk_aoi(prk_geom_pkl, prk_loc_pkl)

    df = pd.read_csv(csv_path, encoding="utf-8-sig", engine="python")
    df = df.rename(columns={k: v for k, v in COLMAP.items() if k in df.columns})

    required_cols = {"source", "x_5179", "y_5179", "poi_name"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Update COLMAP or the CSV.")

    before = len(df)
    df = df[df["source"] == SOURCE_KEEP].copy()
    after = len(df)
    print(f"[filter] kept {after}/{before} rows where source == '{SOURCE_KEEP}'")

    # --- Coordinates ---
    df["x_5179"] = pd.to_numeric(df["x_5179"], errors="coerce")
    df["y_5179"] = pd.to_numeric(df["y_5179"], errors="coerce")
    df = df.dropna(subset=["x_5179", "y_5179"]).copy()

    df["poi_name_clean"] = df["poi_name"].map(normalize_name)
    df = df[df["poi_name_clean"].astype(bool)].copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["x_5179"], df["y_5179"])],
        crs="EPSG:5179",
    ).to_crs("EPSG:4326")

    joined = gpd.sjoin(
        gdf,
        aoi_idx[["gid", "ADM0", "ADM1", "ADM2", "geometry"]],
        how="inner",
        predicate="within",
    )

    out = {}
    for _, r in aoi_idx.iterrows():
        gid = r["gid"]
        block = {
            "ADM0": r["ADM0"],
            "ADM1": r["ADM1"],
            "ADM2": r["ADM2"],
            "production_facilities": wrap_names([]),  # enriched object
        }
        out[gid] = order_block(block)

    if not joined.empty:
        name_lists = (
            joined.groupby("gid")["poi_name_clean"]
            .apply(lambda s: sorted({name for name in s if name}))
        )
        for gid, names in name_lists.items():
            if gid in out:
                out[gid]["production_facilities"] = wrap_names(names)
                out[gid] = order_block(out[gid])

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    total_names = sum(len(blk["production_facilities"]["val"]) for blk in out.values())
    print(f"areas={len(out)} | total_unique_names={total_names}")
    print(f"wrote {out_json}")

if __name__ == "__main__":
    main()
