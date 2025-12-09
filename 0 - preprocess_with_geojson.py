import json
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import pandas as pd
from modules.helper import get_ring_contained_loc

def _std_names(props: dict) -> Tuple[str, str]:
    name1 = props.get('ADM1', props.get('NAME_1'))
    name2 = props.get('ADM2', props.get('NAME_2'))
    if name1 is None or name2 is None:
        raise ValueError("Each feature must have ADM1/NAME_1 and ADM2/NAME_2")
    return str(name1), str(name2)
def _std_names2(props: dict) -> Tuple[str, str]:
    print(props.get('name'))
    return props.get('name').split()[1:]

def _indexing(features: List[dict], ccode: str) -> Dict[int, Tuple[int,int]]:
    """
    Return per-feature indices (adm1_idx, adm2_idx), 1-based.
    ADM1 order: sorted unique ADM1 names.
    ADM2 order: within each ADM1, sorted unique ADM2 names.
    """
    adm1_names = sorted({ _std_names2(f.get('properties', {}))[0] for f in features })
    adm1_map = {name: i+1 for i, name in enumerate(adm1_names)}
    by_adm1 = {a: set() for a in adm1_names}
    for f in features:
        a1, a2 = _std_names2(f.get('properties', {}))
        by_adm1[a1].add(a2)
    adm2_maps = {a1: {name: j+1 for j, name in enumerate(sorted(by_adm1[a1]))} for a1 in adm1_names}
    per_feat = {}
    for i, f in enumerate(features):
        a1, a2 = _std_names2(f.get('properties', {}))
        per_feat[i] = (adm1_map[a1], adm2_maps[a1][a2])
    return per_feat

import json
import pandas as pd
from shapely.geometry import shape as shp_shape
try:
    # Shapely 2.x
    from shapely import make_valid
except ImportError:
    # Shapely 1.8 fallback
    from shapely.validation import make_valid
from pyproj import Geod

geod = Geod(ellps="WGS84")


import json
import pandas as pd
from shapely.geometry import shape as shp_shape
# geod: pyproj.Geod 객체, make_valid: shapely.validation.make_valid 가정

def build_from_geojson(path: str, ccode: str='PRK'):
    with open(path, 'r', encoding='utf-8') as f:
        fc = json.load(f)
    if fc.get('type') != 'FeatureCollection':
        raise ValueError("Expected FeatureCollection")
    feats = fc.get('features', [])
    if not feats:
        raise ValueError("No features found")

    con_loc_dict, geom_dict = {}, {}
    records, out_feats = [], []

    for feat in feats:
        props = feat.get('properties', {}) or {}
        # a1_name, a2_name = _std_names2(props)
        # areaid = props.get('name')
        a1_name,a2_name = props.get('adm1'),props.get('adm2')
        areaid = props.get('target_id') +' ' + a1_name + ' ' + a2_name
        geom_geojson = feat.get('geometry')
        if not geom_geojson:
            continue

        # repr_loc: GeoJSON dict 그대로 (요청사항 유지)
        repr_loc = get_ring_contained_loc(geom_geojson)

        # dict -> Shapely geometry로 변환 + 유효화
        shp = shp_shape(geom_geojson)
        shp_valid = make_valid(shp)

        # weight: 면적(km^2)
        try:
            area_m2, _ = geod.geometry_area_perimeter(shp_valid)
            weight_km2 = abs(area_m2) * 1e-6
        except Exception:
            weight_km2 = None

        out_props = {
            'ADM0': ccode,
            'ADM1': a1_name,
            'ADM2': a2_name,
            'areaid': areaid,
            'repr_loc': repr_loc,   # GeoJSON 기반
            'weight': weight_km2    # km^2
        }

        key = areaid
        con_loc_dict[key] = out_props

        # 🔁 여기 변경: dict가 아니라 Shapely geometry를 저장
        geom_dict[key] = shp_valid

        out_feats.append({
            'type': 'Feature',
            'properties': out_props,
            'geometry': geom_geojson  # 출력 FC는 원본 GeoJSON 유지
        })
        records.append(out_props)

    out_fc = {'type': 'FeatureCollection', 'features': out_feats}
    df = pd.DataFrame.from_records(records, columns=['ADM0','ADM1','ADM2','areaid','repr_loc','weight'])
    return con_loc_dict, geom_dict, out_fc, df


def run_geojson_only(input_geojson: str, out_dir: str, basename: str, ccode: str='PRK'):
    input_geojson = str(Path(input_geojson))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    con_loc_dict, geom_dict, out_fc, df = build_from_geojson(input_geojson, ccode=ccode)

    # Write files to match: {basename}.geojson, {basename}.pkl, {basename}_loc.pickle, {basename}_geom.pickle
    gj_path = out_dir / f"{basename}.geojson"
    with open(gj_path, 'w', encoding='utf-8') as f:
        json.dump(out_fc, f, ensure_ascii=False)

    pkl_path = out_dir / f"{basename}.pkl"
    df.to_pickle(pkl_path)

    loc_path = out_dir / f"{basename}_loc.pickle"
    geom_path = out_dir / f"{basename}_geom.pickle"
    with open(loc_path, 'wb') as f:
        pickle.dump(con_loc_dict, f)
    with open(geom_path, 'wb') as f:
        pickle.dump(geom_dict, f)

    return {'geojson': str(gj_path), 'pkl': str(pkl_path), 'loc_pickle': str(loc_path), 'geom_pickle': str(geom_path)}

if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser(description='GeoJSON-only, PRK-style outputs')
    ap.add_argument('--input_geojson')
    ap.add_argument('--out_dir', default='./outputs_geojson_only')
    ap.add_argument('--basename', default='PRK')
    ap.add_argument('--ccode', default='PRK')
    args = ap.parse_args()
    outs = run_geojson_only(args.input_geojson, out_dir=args.out_dir, basename=args.basename, ccode=args.ccode)
    print(json.dumps(outs, ensure_ascii=False, indent=2))
