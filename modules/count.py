import utils
import json
from pathlib import Path

from .helper import _point2adm2imgpathlist, point2adm2meta, get_ccode_loc
from .segmentation.segmentation import get_segments

_CONFIG = utils.read_config()

def get_landcover_ratio(loc, target_class, config, zoom_level=None, proxy_dir=None):
    """
    Count the pixels in the area of the target class in the location image
    
    Args:
        target_class: The class of target object
        loc: The location of the interest
        zoom_level: Zoom level of the image
        proxy_dir: The proxy dir to save the intermediate results
    
    Return:
        Number of pixels in the area of target class objects in the location image
    """
    # timeline =32645
    
    timeline = config.arcgis.timeline
    proxy_dir = config.path.proxy_dir
    # if proxy_dir is None:
    #     proxy_dir = _CONFIG.path.proxy_dir
    ccode, adm1_name, adm2_name, areaid, _ = point2adm2meta(loc, config)
    proxy_dir = Path(proxy_dir)
    seg_path = proxy_dir / f'{ccode}_segmentation_{timeline}.json'
    loc_data = get_ccode_loc(ccode)
    assert len(loc_data)>0
    loc_weight = loc_data[areaid]['weight']

    if seg_path.exists():
        with open(seg_path, 'r') as file:
            data = json.load(file)
           
        if areaid in data.keys() and 'segmentation' in data[areaid].keys():
            target_area_info = data[areaid]['segmentation'][target_class]
            result_dict = {'segmentation': target_area_info}
            return result_dict
    else:
        data = {}
    
    data[areaid] = {}
    img_path_list = _point2adm2imgpathlist(loc,config, zoom_level, timeline=timeline)
    area_dict = get_segments(img_path_list)

    for each_class in area_dict.keys():
        area_dict[each_class]['type'] = 'ratio'
        area_dict[each_class]['weight'] = loc_weight
    
    data[areaid]["ADM0"] = ccode
    data[areaid]["ADM1"] = adm1_name
    if adm2_name is not None:
        data[areaid]["ADM2"] = adm2_name
    data[areaid]['segmentation'] = area_dict
        
    with open(seg_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    result_dict = {'segmentation': area_dict[target_class]}
    return result_dict
def get_landcover_ratio_indep( ):
    """
    Count the pixels in the area of the target class in the location image
    
    Args:
        target_class: The class of target object
        loc: The location of the interest
        zoom_level: Zoom level of the image
        proxy_dir: The proxy dir to save the intermediate results
    
    Return:
        Number of pixels in the area of target class objects in the location image
    """
    proxy_dir = "/home/donggyu/GELT/v4_satelite_img/"
    seg_path = proxy_dir + f'HU_19_sample_segmentation.json'
    data = {}
    import glob
    data['test'] = {}
    img_path_list  = glob.glob("/home/donggyu/GELT/v4_satelite_img/HU_19_sample/"+'*_*.png')
    area_dict = get_segments(img_path_list)

    for each_class in area_dict.keys():
        area_dict[each_class]['type'] = 'ratio'
        area_dict[each_class]['weight'] = 'test'
    
    data['test']["ADM0"] = 'test'
    data['test']["ADM1"] = 'test'

    data['test']["ADM2"] = 'test'
    data['test']['segmentation'] = area_dict
        
    with open(seg_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    # result_dict = {'segmentation': area_dict[target_class]}
    # return result_dict
import glob
def get_landcover_ratio_per_image():
    """
    img_path_list에 있는 각 이미지를 독립적으로 처리하여
    클래스별 픽셀 비율/개수를 추출한 뒤 JSON으로 저장
    """
    proxy_dir = "/home/donggyu/GELT/v4_satelite_img/"
    seg_path = proxy_dir + 'HU_arcgis_segmentation_per_image.json'
    # 5699_9179
    # 5699_9190
    # 5785_9112
    # 5841_8932
    # 1) 이미지 경로 리스트 만들기
    img_path_list = glob.glob("/home/donggyu/GELT/v4_satelite_img/HU_sentinel/arcgis/*_*.png")
    # img_path_list = 

    # 2) 결과를 담을 dict 생성
    data = {"test": {}}

    # 3) 이미지별로 반복 처리
    for img_path in img_path_list:
        # get_segments는 [경로리스트]를 인자로 받으므로, 단일 이미지에도 리스트로 감싸서 호출
        area_dict = get_segments([img_path])

        # area_dict 구조: { "class1": {...}, "class2": {...}, ... }
        # 여기에 추가 메타정보(타입, weight 등)를 붙이고 싶으면:
        for cls, info in area_dict.items():
            info['type']   = 'ratio'
            info['weight'] = 'test'

        # 파일 이름만 키로 쓰고 싶다면 os.path.basename(img_path)을 사용
        img_key = img_path.split('/')[-1]  # 또는 os.path.basename(img_path)

        # 결과 저장
        data['test'][img_key] = {
            "ADM0": "test",    # 필요에 따라 실제 ADM 정보를 여기에 넣으세요
            "ADM1": "test",
            "ADM2": "test",
            "segmentation": area_dict
        }

    # 4) JSON으로 dump
        with open(seg_path, 'w') as f:
            json.dump(data, f, indent=4)

    return data  # 필요에 따라 반환



def get_landuse_sum(loc, target_class, config, zoom_level=None, proxy_dir=None):
    """
    GeoChat module calculates the total sum of each classes.
    
    Args:
        target_class: The class of target object
        loc: The location of the interest
        zoom_level: Zoom level of the image
        proxy_dir: The proxy dir to save the intermediate results
    
    Return:
        sum of probabilities of each classes within the target location
    """
    # timeline =32645
    # config = utils.read_config()
    timeline = config.arcgis.timeline
    proxy_dir = config.path.proxy_dir
    # if proxy_dir is None:
    #     proxy_dir = _CONFIG.path.proxy_dir
    ccode, adm1_name, adm2_name, areaid, _ = point2adm2meta(loc, config)
    proxy_dir = Path(proxy_dir)
    seg_path = proxy_dir / f'{ccode}_landuse_{timeline}.json'
    print(seg_path)
    loc_data = get_ccode_loc(ccode)
    assert len(loc_data)>0
    loc_weight = loc_data[areaid]['weight']

    if seg_path.exists():
        with open(seg_path, 'r') as file:
            data = json.load(file)

        if areaid in data.keys() and 'landuse' in data[areaid].keys():
            
            # if 'img_count' in data[areaid].keys() and data[areaid]['img_count']['val'] <=5:
            #     # print("hji")
            #     result_dict = {'segmentation': None}
            #     return result_dict
            target_area_info ={}
            percentage = data[areaid]['landuse']['Landuse_Sum']['val'][target_class]/data[areaid]['landuse']['Landuse_Average']['weight'] *100
            target_area_val = (str(round(percentage, 1)) +'%', str(round(percentage/100 *loc_weight, 1)) + 'km²')
            target_area_desc= f'The {target_class} area percentage and area estimated from satellite imagery'
            target_area_type= data[areaid]['landuse']['Landuse_Sum']['type']
            target_area_weight= data[areaid]['landuse']['Landuse_Sum']['weight']
            target_area_info['val'] = target_area_val
            target_area_info['desc'] = target_area_desc
            target_area_info['type'] = target_area_type
            target_area_info['weight'] = target_area_weight
            result_dict = {'segmentation': target_area_info}
            return result_dict
        else:
            return {}
    else:
        data = {}
    

    return result_dict

def get_landuse_sum_original(loc, target_class, config, zoom_level=None, proxy_dir=None):
    """
    GeoChat module calculates the total sum of each classes.
    
    Args:
        target_class: The class of target object
        loc: The location of the interest
        zoom_level: Zoom level of the image
        proxy_dir: The proxy dir to save the intermediate results
    
    Return:
        sum of probabilities of each classes within the target location
    """
    # timeline =32645
 # timeline =32645
    # config = utils.read_config()
    timeline = config.arcgis.timeline
    proxy_dir = config.path.proxy_dir
    # if proxy_dir is None:
    #     proxy_dir = _CONFIG.path.proxy_dir
    ccode, adm1_name, adm2_name, areaid, _ = point2adm2meta(loc, config)
    proxy_dir = Path(proxy_dir)
    seg_path = proxy_dir / f'{ccode}_landuse_{timeline}.json'
    print(seg_path)
    loc_data = get_ccode_loc(ccode)
    assert len(loc_data)>0
    loc_weight = loc_data[areaid]['weight']

    if seg_path.exists():
        with open(seg_path, 'r') as file:
            data = json.load(file)
           
        if areaid in data.keys() and 'landuse' in data[areaid].keys():
            target_area_info ={}
            target_area_val = data[areaid]['landuse']['Landuse_Sum']['val'][target_class]
            target_area_desc= f'{target_class} satelite image count'
            target_area_type= data[areaid]['landuse']['Landuse_Sum']['type']
            target_area_weight= data[areaid]['landuse']['Landuse_Sum']['weight']
            target_area_info['val'] = target_area_val
            target_area_info['desc'] = target_area_desc
            target_area_info['type'] = target_area_type
            target_area_info['weight'] = target_area_weight
            result_dict = {'segmentation': target_area_info}
            return result_dict
    else:
        data = {}
    

    return result_dict

