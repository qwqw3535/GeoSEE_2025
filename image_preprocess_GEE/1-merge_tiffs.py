import argparse
import os
import glob
import sys
import subprocess

def parse_arguments(notebook=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='.\\HU_tif_raw\\', help='Source path')
    parser.add_argument('--des', type=str, default='.\\HU_SR_tif_merged\\', help='Destination path')
    if notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    print(args)

    source_path = args.src
    destination_path = args.des
    os.makedirs(destination_path, exist_ok=True)

    output_file = os.path.join(destination_path, 'merged.tif')
    print(f"Output: {output_file}")

    # 재귀적으로 모든 .tif(.tiff) 수집
    alltiffs = []
    alltiffs.extend(glob.glob(os.path.join(source_path, '**', '*.tif'), recursive=True))
    alltiffs.extend(glob.glob(os.path.join(source_path, '**', '*.tiff'), recursive=True))
    # 중복 제거 및 정렬(재현성)
    alltiffs = sorted(set(alltiffs))

    print(f"Found {len(alltiffs)} TIFFs")
    if not alltiffs:
        print("No TIFF files found under the source path. Exiting.")
        sys.exit(0)

    # 환경에 맞게 'gdal_merge' 또는 'gdal_merge.py' 사용
    gdal_merge_cmd = 'gdal_merge'  # 또는 'gdal_merge.py'

    cmd = [
        gdal_merge_cmd,
        '-o', output_file,
        '-co', 'COMPRESS=LZW',
        '-co', 'BIGTIFF=YES',
        '-co', 'PREDICTOR=2',
        '-co', 'TILED=YES',
    ] + alltiffs

    print("Running merge command...")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        # gdal_merge가 아니라 gdal_merge.py만 있는 환경일 수 있음
        cmd[0] = 'gdal_merge.py'
        subprocess.run(cmd, check=True)

    print("All Finished")
