import os, glob
import argparse
import subprocess
import shutil
import numpy as np
import gc
from osgeo import gdal

# 3) GDAL 캐시 제한 (256MB)
gdal.SetCacheMax(256 * 1024 * 1024)
os.environ.setdefault("GDAL_CACHEMAX", "256")  # 서브프로세스용(예: gdal_translate)

def parse_arguments(notebook=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='HU_raw_tif/', help='Source path (files 모드: TIF들이 있는 폴더, dirs 모드: 하위폴더들이 모여있는 루트 폴더)')
    parser.add_argument('--des', type=str, default='HU_SR_tif_unblurry/', help='Destination path')
    parser.add_argument('--pmin', type=float, default=2.0, help='Lower percentile (e.g., 2)')
    parser.add_argument('--pmax', type=float, default=98.0, help='Upper percentile (e.g., 98)')
    parser.add_argument('--dtype', type=str, default='Byte',
                        choices=['Byte','UInt16','Int16','UInt32','Int32','Float32','Float64'],
                        help='Output data type for gdal_translate')
    parser.add_argument('--pattern', type=str, default='*.tif', help='Glob pattern for input files (예: "*.tif", "*_B*.tif")')
    parser.add_argument('--skip_existing', action='store_true', help='Skip if output exists')
    parser.add_argument('--recursive', action='store_true', help='files 모드에서 하위 폴더까지 재귀적으로 검색')
    parser.add_argument('--mode', type=str, default='files', choices=['files', 'dirs'],
                        help='files: src 폴더의 tif들을 처리 / dirs: src의 각 하위폴더를 각각 처리(출력은 des/<하위폴더>/)')
    if notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args


def robust_percentiles_per_band(path, pmin=2.0, pmax=98.0):
    """
    전체 이미지를 한 번에 읽되, 메모리 복사 최소화:
    - float32로 단 한 번 업캐스트
    - NoData는 in-place로 NaN 지정
    - np.nanpercentile로 퍼센타일 (추가 대용량 복사 없음)
    """
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f'Failed to open {path}')

    results = []
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b)
        nodata = band.GetNoDataValue()

        # float32로 읽기 (정수→float 변환은 이 한 번만)
        arr = band.ReadAsArray().astype(np.float32, copy=False)
        if arr is None:
            raise RuntimeError(f'Failed to read band {b} from {path}')

        # NoData -> NaN (대규모 복사 없이 in-place)
        if nodata is not None:
            arr[arr == nodata] = np.nan

        # NaN 제외 퍼센타일
        lo = np.nanpercentile(arr, pmin)
        hi = np.nanpercentile(arr, pmax)

        # 안전장치
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            mn2, mx2 = band.ComputeRasterMinMax(True)
            if mn2 == mx2:
                lo, hi = mn2 - 0.5, mx2 + 0.5
            else:
                lo, hi = mn2, mx2

        results.append((float(lo), float(hi)))

        # 메모리 즉시 해제
        del arr
        gc.collect()

    ds = None
    return results


def build_gdal_translate_cmd(in_path, out_path, dtype, band_scales):
    cmd = ['gdal_translate', '-ot', dtype]
    for idx, (src_min, src_max) in enumerate(band_scales, start=1):
        cmd += ['-b', str(idx), '-scale', f'{src_min}', f'{src_max}', '0', '255']
    cmd += [in_path, out_path]
    return cmd


def process_one_file(input_file, output_file, dtype, pmin, pmax):
    # 퍼센타일 계산
    scales = robust_percentiles_per_band(input_file, pmin, pmax)
    # gdal_translate 실행
    cmd = build_gdal_translate_cmd(input_file, output_file, dtype, scales)
    env = os.environ.copy()
    env.setdefault("GDAL_CACHEMAX", "256")
    subprocess.run(cmd, check=True, env=env)


def list_files_files_mode(src, pattern, recursive):
    if recursive:
        # ** 패턴으로 재귀
        search_pattern = os.path.join(src, '**', pattern)
        files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(src, pattern)
        files = glob.glob(search_pattern)
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    return files


def list_dirs(src):
    # src의 즉시 하위 폴더만
    return sorted([d for d in (os.path.join(src, x) for x in os.listdir(src))
                   if os.path.isdir(d)])


if __name__ == '__main__':
    args = parse_arguments()

    if shutil.which('gdal_translate') is None:
        raise SystemExit('gdal_translate not found in PATH.')

    os.makedirs(args.des, exist_ok=True)

    if args.mode == 'files':
        # 기존 동작: (옵션) 재귀 탐색 지원
        files = list_files_files_mode(args.src, args.pattern, args.recursive)
        print(f"[files mode] Found {len(files)} files under: {args.src}")
        if not files:
            print('No files found.')

        for input_file in files:
            rel = os.path.relpath(input_file, start=args.src)  # 구조 보존을 위해 상대경로
            out_path = os.path.join(args.des, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if args.skip_existing and os.path.exists(out_path):
                print(f'{rel} exists, skip.')
                continue

            try:
                process_one_file(input_file, out_path, args.dtype, args.pmin, args.pmax)
                print(f'{rel} Finished. (P{args.pmin}–P{args.pmax})')
            except Exception as e:
                print(f'Error on {rel}: {e}')

    else:  # args.mode == 'dirs'
        # 새 동작: src의 각 하위폴더를 "각각 한 번에" 처리
        subdirs = list_dirs(args.src)
        print(f"[dirs mode] Found {len(subdirs)} subdirectories in: {args.src}")
        if not subdirs:
            print('No subdirectories found.')

        for d in subdirs:
            dir_name = os.path.basename(d.rstrip(os.sep))
            files = sorted(glob.glob(os.path.join(d, args.pattern)))
            print(f'  - {dir_name}: {len(files)} files')
            out_dir = os.path.join(args.des, dir_name)
            os.makedirs(out_dir, exist_ok=True)

            for input_file in files:
                file_name = os.path.basename(input_file)
                out_path = os.path.join(out_dir, file_name)

                if args.skip_existing and os.path.exists(out_path):
                    print(f'    {dir_name}/{file_name} exists, skip.')
                    continue

                try:
                    process_one_file(input_file, out_path, args.dtype, args.pmin, args.pmax)
                    print(f'    {dir_name}/{file_name} Finished. (P{args.pmin}–P{args.pmax})')
                except Exception as e:
                    print(f'    Error on {dir_name}/{file_name}: {e}')

    print('Finished')
