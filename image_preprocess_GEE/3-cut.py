#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from osgeo import gdal
from osgeo_utils import gdal2tiles
import subprocess
import shutil

def parse_arguments(notebook: bool=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--z', type=str, default='14', help="Zoom levels: '2-5', '10-', or '10'")
    parser.add_argument('--p', type=int, default=15, help='Parallel workers (CPU cores)')
    parser.add_argument('--size', type=int, default=256, help='Tile size (256/512/1024)')
    parser.add_argument('--src', type=str, default='HU_SR_tif_unblurry', help='Source directory')
    parser.add_argument('--des', type=str, default='HU_SR_Tiles', help='Destination directory')
    parser.add_argument('--resume', action='store_true', help='Resume incomplete tiling')
    parser.add_argument('--overwrite', action='store_true', help='Delete per-TIF output dirs before running')
    parser.add_argument('--geodetic', action='store_true',
                        help='Make EPSG:4326 tiles (gdal2tiles -p geodetic). Default: WebMercator tiles.')
    parser.add_argument('--prewarp_3857', action='store_true',
                        help='Pre-warp to EPSG:3857 before tiling (recommended for stability). Ignored if --geodetic.')
    parser.add_argument('--keep_tmp', action='store_true', help='Keep temporary warped files for debugging')
    parser.add_argument('--recursive', action='store_true', help='Recursively search for .tif under --src')
    if notebook:
        return parser.parse_args([])
    return parser.parse_args()

def warp_to_3857(src_tif: Path, out_dir: Path) -> Path:
    """Pre-warp source to EPSG:3857 with alpha; returns path to warped GeoTIFF."""
    out_tif = out_dir / f"{src_tif.stem}_3857.tif"
    if out_tif.exists():
        return out_tif
    warp_opts = gdal.WarpOptions(
        dstSRS="EPSG:3857",
        resampleAlg="bilinear",
        dstAlpha=True,
        multithread=True
    )
    print(f"[warp] {src_tif.name} -> {out_tif.name} (EPSG:3857)")
    ds = gdal.Warp(destNameOrDestDS=str(out_tif), srcDSOrSrcDSTab=str(src_tif), options=warp_opts)
    if ds is None:
        raise RuntimeError(f"gdal.Warp failed for {src_tif}")
    ds = None
    return out_tif

def run_gdal2tiles(
    tif_path: Path,
    out_dir: Path,
    z: str,
    tilesize: int,
    inner_procs: int,
    resume: bool,
    geodetic: bool,
    prewarp_3857: bool,
    keep_tmp: bool
):
    src_for_tiles = tif_path
    tmp_to_delete = None
    if not geodetic and prewarp_3857:
        src_for_tiles = warp_to_3857(tif_path, out_dir)
        if not keep_tmp:
            tmp_to_delete = src_for_tiles

    cmd = [
        "gdal2tiles.py",
        "--zoom", str(z),
        "--tilesize", str(tilesize),
        "--xyz",
        "--processes", str(inner_procs),
        "--resampling", "bilinear",
        "--webviewer", "none",
        str(src_for_tiles),
        str(out_dir),
    ]
    if resume:
        cmd.append("--resume")
    if geodetic:
        cmd += ["--profile", "geodetic"]

    try:
        cmd_with_alpha = cmd[:]
        cmd_with_alpha.insert(1, "--dstalpha")
        print("[gdal2tiles]", " ".join(cmd_with_alpha))
        subprocess.run(cmd_with_alpha, check=True)
    except subprocess.CalledProcessError:
        print("[warn] --dstalpha 미지원 버전으로 보입니다. --dstalpha 없이 재시도합니다.")
        print("[gdal2tiles]", " ".join(cmd))
        subprocess.run(cmd, check=True)

    if tmp_to_delete and Path(tmp_to_delete).exists():
        try:
            Path(tmp_to_delete).unlink()
        except Exception as e:
            print(f"[warn] tmp 삭제 실패: {tmp_to_delete} ({e})")

def find_tifs(src_dir: Path, recursive: bool):
    """
    src_dir 아래의 .tif/.TIF 파일을 재귀/비재귀로 탐색.
    """
    if recursive:
        it = src_dir.rglob("*")
    else:
        it = src_dir.glob("*")
    tifs = [p for p in it if p.is_file() and p.suffix.lower() == ".tif"]
    tifs.sort()
    return tifs

def tiles_outdir_for(src_dir: Path, des_dir: Path, tif_path: Path) -> Path:
    """
    원본 상대 경로 구조를 보존하여 출력 디렉토리 결정.
    src/A/B/c.tif -> des/A/B/c/
    """
    rel = tif_path.relative_to(src_dir)
    return des_dir / rel.parent / tif_path.stem

def main():
    args = parse_arguments()
    src_dir = Path(args.src)
    des_dir = Path(args.des)
    des_dir.mkdir(parents=True, exist_ok=True)

    tif_list = find_tifs(src_dir, args.recursive)
    if not tif_list:
        print(f"No .tif files found in {src_dir.resolve()} (recursive={args.recursive})")
        return

    n_files = len(tif_list)
    # 여러 파일이면 파일 단위 병렬 + gdal2tiles 내부 프로세스는 1
    if n_files > 1:
        file_workers = min(args.p, n_files)
        inner_procs = 1
        print(f"[Multi-file] files={n_files} | file_workers={file_workers} | inner_processes={inner_procs}")
        futures = []
        with ProcessPoolExecutor(max_workers=file_workers) as ex:
            for tif_path in tif_list:
                out_dir = tiles_outdir_for(src_dir, des_dir, tif_path)
                if args.overwrite and out_dir.exists():
                    shutil.rmtree(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                futures.append(
                    ex.submit(
                        run_gdal2tiles,
                        tif_path,
                        out_dir,
                        args.z,
                        args.size,
                        inner_procs,
                        args.resume,
                        args.geodetic,
                        args.prewarp_3857,
                        args.keep_tmp
                    )
                )
            for f in as_completed(futures):
                f.result()
    else:
        tif_path = tif_list[0]
        out_dir = tiles_outdir_for(src_dir, des_dir, tif_path)
        if args.overwrite and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inner_procs = max(1, args.p)
        print(f"[Single-file] inner_processes={inner_procs}")
        run_gdal2tiles(
            tif_path,
            out_dir,
            args.z,
            args.size,
            inner_procs,
            args.resume,
            args.geodetic,
            args.prewarp_3857,
            args.keep_tmp
        )

    print("Finished")

if __name__ == "__main__":
    main()
