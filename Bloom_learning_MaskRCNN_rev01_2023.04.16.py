# !pip install scikit-image
import re
import torch
print(torch.__version__)
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
# 런타임->런타임 다시 시작 후 아래 수행. 
from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm
pd.set_option('display.max_colwidth', 300)
import matplotlib.pyplot as plt
import cv2
import pycocotools.mask as maskUtils
# 학습과 검증용 image id 데이터들을 추출. 
from sklearn.model_selection import train_test_split
import json
import shutil
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from pycocotools.coco import COCO
from mmdet.apis import set_random_seed
import os.path as osp
import panopticapi
from panopticapi.evaluation import VOID
from panopticapi.utils import id2rgb

def show_image_masks(image_file_name, mask_file_list, cols=5):
    figure, axs = plt.subplots(nrows=1, ncols=cols, figsize=(16, 12))
    for i in range(cols):
        im_name = image_file_name if i ==0 else mask_file_list[i-1]
        im_array = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
        print('{0:} shape:{1:}'.format('image' if i == 0 else 'mask', im_array.shape))
        axs[i].imshow(im_array)
        axs[i].set_title('image' if i == 0 else 'mask'+str(i))
        


def get_imagename_list(data_root_dir):
    imagename_list = []
    imageid_list = []
    # data_root_dir 밑에 있는 모든 image id 디렉토리의 서브 디렉토리 중 images 디렉토리 밑에 있는 image 파일 명을 절대 경로로 추출
    # data_root_dir -> image id dir -> images -> image 파일명 
    for dir in sorted(next(os.walk(data_root_dir))[1]):
        subdirs = os.path.join(data_root_dir, dir)
        if 'images' in subdirs:
            image_dir = subdirs
            # print(image_dir)
            for imagename in sorted(next(os.walk(image_dir))[2]):
                # print(imagename.replace('.JPG', '').replace('.jpg', ''))
                imagename_list.append(os.path.join(image_dir, imagename))
                imageid_list.append(int(imagename.replace('.JPG', '').replace('.jpg', '').replace('.png', '')))
    
    return imageid_list, imagename_list

# 특정 image id 디렉토리 밑에 있는 mask 파일명을 절대 경로로 모두 추출.
# image_id dir -> masks -> 여러 mask 파일 명 
def get_maskname_list(data_root_dir, idx):
    mask_dir = os.path.join(data_root_dir, 'masks')
    maskname_list_list = []
    maskname_list = []
    first = []
    second = []
    third = []
    fourth = []
    fifth = []
    sixth = []
    seventh = []
    eighth = []
    ninth = []
    tenth = []
    eleventh = []
    twelfth = []
    for mask_filename in next(os.walk(mask_dir))[2]:
        if 'png' in mask_filename:
            # print(mask_filename)
            image_id = int(re.split('-', mask_filename)[1])
            if idx == image_id:
                if labels_to_names_seq[0] in mask_filename:                    
                    first.append(os.path.join(mask_dir, mask_filename))
                elif labels_to_names_seq[1] in mask_filename:                    
                    second.append(os.path.join(mask_dir, mask_filename))
                elif labels_to_names_seq[2] in mask_filename:                    
                    third.append(os.path.join(mask_dir, mask_filename))
                elif labels_to_names_seq[3] in mask_filename:
                    fourth.append(os.path.join(mask_dir, mask_filename))
                elif labels_to_names_seq[4] in mask_filename:
                    fifth.append(os.path.join(mask_dir, mask_filename))
                elif labels_to_names_seq[5] in mask_filename:
                    sixth.append(os.path.join(mask_dir, mask_filename))
                elif labels_to_names_seq[6] in mask_filename:
                    seventh.append(os.path.join(mask_dir, mask_filename))    
                elif labels_to_names_seq[7] in mask_filename:
                    eighth.append(os.path.join(mask_dir, mask_filename))
                elif labels_to_names_seq[9] in mask_filename:
                    tenth.append(os.path.join(mask_dir, mask_filename))    
                elif labels_to_names_seq[10] in mask_filename:
                    eleventh.append(os.path.join(mask_dir, mask_filename))    
                elif labels_to_names_seq[11] in mask_filename:
                    twelfth.append(os.path.join(mask_dir, mask_filename))
                # bloom은 제일 마지막에 수집
                elif labels_to_names_seq[8] in mask_filename:
                    ninth.append(os.path.join(mask_dir, mask_filename))
          
            else:
                continue      
    maskname_list.append([first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, twelfth])                              
    return maskname_list[0]

def make_meta_df(dataset_dir):
    # 모든 image id들의 list를 가져옴.
    # 모든 image 절대경로들의 list를 가져옴.
    imageid_list, imagename_list = get_imagename_list(dataset_dir)
    # 개별 image당 모든 mask 절대 경로를 가져옴. 각 행이 경로모음list가 됨
    maskname_list_list = []
    for imageid in imageid_list:
        maskname_list = get_maskname_list(dataset_dir, idx=imageid)
        maskname_list_list.append(maskname_list)
    
    print(len(imageid_list), len(imagename_list), len(maskname_list))
    
    
    mata_df = pd.DataFrame({'image_id' : imageid_list,
                            'image_name' : imagename_list,
                            'maskname_list' : maskname_list_list
                            })
    return mata_df

# bounding box 정보를 polygon에서 추출. 
def get_bbox(segm):
    x_min = float('inf')
    y_min = float('inf')
    x_max = 0
    y_max = 0
    # print(segm)
    #segmentation polygon정보로 bounding box 정보 추출. 
    # cv2.boundingRect : contour에 외접하는 똑바로 세워진 사각형 좌표
    x, y, w, h = cv2.boundingRect(segm)
    x_b = x + w
    y_b = y + h
    # 최소 0보다 크고, 무한대 보다는 좌표값이 작아야 함.
    x_min = min(x_min, x)
    y_min = min(y_min, y)
    x_max = max(x_max, x_b)
    y_max = max(y_max, y_b)
        
    # 좌상단 좌표와, width, height 반환. 
    return x_min, y_min, x_max - x_min, y_max - y_min

# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/pipelines/loading.py def _poly2mask(self, mask_ann, img_h, img_w):
import pycocotools.mask as maskUtils

def check_polygons(mask_filepath, polygons, img_h, img_w):
    try:
        rles = maskUtils.frPyObjects(polygons, img_h, img_w)
        rle = maskUtils.merge(rles)
        # print(rle)
    except Exception as e:
        print('##### 오류 polygon 발생 #####:',  e,polygons)
        print('오류 mask file명:', mask_filepath)
        
# mask image를 기반으로 segmentation polygon과 bbox 정보를 추출하는 로직 함수화
def get_annotation_info(mask_filename):
    
    bboxs = []
    
    # mask_filepath = os.path.join(mask_dir, mask_filename)
    mask_filepath = mask_filename
    # print(mask_filepath)
    mask_array = cv2.imread(mask_filepath)
    mask_array = np.where(mask_array > 0, 255, mask_array)
    contours, hierarchy = cv2.findContours(mask_array[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 좌표 2개초과인 경우만 좌표수집
    # contour → mask 재변환시 좌표개수가 2보다 작으면 다각형이 안 만들어지므로 문제가 됨
    # chech_polygons를 이용하면 확인가능
    # segm = [contour for contour in contours if contour.shape[0] > 2]  
    # 블룸표면크랙의 경우 : 크랙이 너무 많아 학습이 안되므로 라벨 개수 감소를 위해 추가 필터링 실시
    segm = [contour for contour in contours if contour.shape[0] > 23]  
    
    # polygons = [s.ravel().tolist() for s in contours]
    polygons = [s.ravel().tolist() for s in segm] # 실제 필요코드
    check_polygons(mask_filepath, polygons, mask_array.shape[0], mask_array.shape[1])
    # 만약 polygons가 Null이면 segmentation과 bbox annotation을 None으로 반환.
    if polygons == []:
        return None, None
    # polygons가 Null이 아니면 정상적으로 segmentation과 bbox annotation 반환. 
    else:
        for num in range(len(polygons)):
            xmin, ymin, bwidth, bheight = get_bbox(segm[num])  
            bboxs.append([xmin, ymin, bwidth, bheight])
             
        return polygons, bboxs

# mmdetection config에 맞추기 위해 nucleus 이미지 파일을 별도의 디렉토리로 이동하는 로직 추가 
def convert_bloom_to_coco(data_root_dir, image_ids, out_file, img_copy_dir, meta_df, labels_to_names_seq):
    
    images = []
    annotations = []
    categories = []
    obj_index = 0
    
    # image 정보를 담아서 images list에 추가.
    for index, image_id in enumerate(image_ids):
        
        # image_id_dir = os.path.join(data_root_dir, image_id)
        # image_path = os.path.join(image_id_dir, 'images/' + image_id + '.JPG')
        
        image_path = os.path.join(data_root_dir, 'images/' + str(image_id) + '.jpg')
        print(image_path)
        file_name = str(image_id) + '.jpg'
        height, width = cv2.imread(image_path).shape[0:2]
        # 개별 image의 dict 정보 생성
        image_info = dict(file_name=file_name,
                          height=height,
                          width=width,
                          id=str(image_id))
        # 개별 image dict 정보를 images list에 추가. 
        images.append(image_info)
        
        # 이미지를 특정 디렉토리 밑으로 모일 수 있도록 copy
        shutil.copy(image_path, os.path.join(img_copy_dir, file_name))
        
        # 개별 image에 있는 여러 mask 이미지 파일을 segmentation, bbox로 변환하여 annotation dict 정보 생성.
        mask_dir = os.path.join(data_root_dir, 'masks')
        # mask_filename_list = next(os.walk(mask_dir))[2]
        # 개별 image에 있는 여러 mask 이미지 파일을 기반으로 annotation dict 생성.
        for category_id in range(len(labels_to_names_seq)):
            for maskname_list in meta_df['maskname_list']:
                if not maskname_list[category_id]:
                    continue
                for mask_filename in maskname_list[category_id]:                    
                    if image_id != int(re.split('-',re.split('/', mask_filename)[2])[1]):
                        continue
                    else:
                        # 개별 mask 파일에서 polygon list와 bbbox list를 계산하여 반환.
                        # segmentation, bbox = get_annotation_info(mask_dir, mask_filename)
                        segmentation, bbox = get_annotation_info(mask_filename)
                        
                        for num in range(len(segmentation)):
                            # 만일 segmentation이 None이면 coco 데이터로 만들지 않음. 
                            if segmentation[num] is None:
                                print(f'segmentation[{int(num)}] is None')
                                continue  
                            elif not segmentation[num]:
                                print(f'segmentation[{int(num)}] is empty_list')
                                continue   
                            annotation = dict(segmentation=[segmentation[num]], # segmentation좌표는 반드시 이중list로 처리되어야 딥러닝학습이 가능함!!
                                            area= bbox[num][2] * bbox[num][3], # 굳이 area를 계산할 필요가 없어서 0처리
                                            iscrowd = 0,
                                            bbox=bbox[num],
                                            category_id=category_id, # class를 의미함
                                            image_id=str(image_id),
                                            id = obj_index)
                            # 계산된 annotation dict 정보를 annotations list에 추가.
                            annotations.append(annotation)
                            # print(f'annotations 추가')
                            # object 고유 id 증가.
                            obj_index += 1                      
                        
                        print('category_id:', labels_to_names_seq[category_id], 'obj_index:', obj_index, 'image id:', image_id, 'annotation is done')
        index = index + 1
        print(f'전체 {len(image_ids)} 중 {index} Convert 완료')         
        coco_format_json = dict(
            images = images,
            annotations = annotations,
            categories = [{'id': i, 'name':labels_to_names_seq[i]} for i in range(len(labels_to_names_seq))]
        )
        
    # json 파일로 출력. 
    #mmcv.dump(coco_format_json, out_file)
    with open(out_file, 'w') as json_out_file:
        json.dump(coco_format_json, json_out_file)

################### Crack 이진분류 COCO json 파일 생성 ##################
# mmdetection config에 맞추기 위해 nucleus 이미지 파일을 별도의 디렉토리로 이동하는 로직 추가 
def convert_billet_to_coco_crack_one_class(data_root_dir, image_ids, out_file, img_copy_dir, meta_df, labels_to_names_seq):
    
    images = []
    annotations = []
    categories = []
    obj_index = 0
    
    # image 정보를 담아서 images list에 추가.
    for index, image_id in enumerate(image_ids):
        
        # image_id_dir = os.path.join(data_root_dir, image_id)
        # image_path = os.path.join(image_id_dir, 'images/' + image_id + '.JPG')
        
        image_path = os.path.join(data_root_dir, 'images/' + str(image_id) + '.JPG')
        print(image_path)
        file_name = str(image_id) + '.jpg'
        height, width = cv2.imread(image_path).shape[0:2]
        # 개별 image의 dict 정보 생성
        image_info = dict(file_name=file_name,
                          height=height,
                          width=width,
                          id=str(image_id))
        # 개별 image dict 정보를 images list에 추가. 
        images.append(image_info)
        
        # 이미지를 특정 디렉토리 밑으로 모일 수 있도록 copy
        shutil.copy(image_path, os.path.join(img_copy_dir, file_name))
        
        # 개별 image에 있는 여러 mask 이미지 파일을 segmentation, bbox로 변환하여 annotation dict 정보 생성.
        mask_dir = os.path.join(data_root_dir, 'masks')
        # mask_filename_list = next(os.walk(mask_dir))[2]
        # 개별 image에 있는 여러 mask 이미지 파일을 기반으로 annotation dict 생성.
        # Crack 이진분류만 json 파일로 저장하도록 유도
        
        # for category_id in range(len(labels_to_names_seq)):
        for category_id in range(1): 
            for maskname_list in meta_df['maskname_list']:
                if not maskname_list[category_id]:
                    continue
                for mask_filename in maskname_list[category_id]:                    
                    if image_id != int(re.split('-',re.split('/', mask_filename)[2])[3]):
                        continue
                    else:
                        # 개별 mask 파일에서 polygon list와 bbbox list를 계산하여 반환.
                        # segmentation, bbox = get_annotation_info(mask_dir, mask_filename)
                        segmentation, bbox = get_annotation_info(mask_filename)
                        
                        for num in range(len(segmentation)):
                            # 만일 segmentation이 None이면 coco 데이터로 만들지 않음. 
                            if segmentation[num] is None:
                                print(f'segmentation[{int(num)}] is None')
                                continue  
                            elif not segmentation[num]:
                                print(f'segmentation[{int(num)}] is empty_list')
                                continue   
                            annotation = dict(segmentation=[segmentation[num]], # segmentation좌표는 반드시 이중list로 처리되어야 딥러닝학습이 가능함!!
                                            area= bbox[num][2] * bbox[num][3], # 굳이 area를 계산할 필요가 없어서 0처리
                                            iscrowd = 0,
                                            bbox=bbox[num],
                                            category_id=category_id, # class를 의미함
                                            image_id=str(image_id),
                                            id = obj_index)
                            # 계산된 annotation dict 정보를 annotations list에 추가.
                            annotations.append(annotation)
                            # object 고유 id 증가.
                            obj_index += 1                      
                        print('category_id:', labels_to_names_seq[category_id], 'obj_index:', obj_index, 'image id:', image_id, 'annotation is done')
        index = index + 1
        print(f'전체 {len(image_ids)} 중 {index} Convert 완료')         
        coco_format_json = dict(
            images = images,
            annotations = annotations,
            categories = [{'id': i, 'name':labels_to_names_seq[i]} for i in range(len(labels_to_names_seq))]
        )
        
    # json 파일로 출력. 
    #mmcv.dump(coco_format_json, out_file)
    with open(out_file, 'w') as json_out_file:
        json.dump(coco_format_json, json_out_file)


import numpy as np

# coco data 실습에 사용된 시각화 함수를 그대로 가져옴. 
def get_polygon_xy(ann_seg):
    polygon_x = [x for index, x in enumerate(ann_seg) if index % 2 == 0]
    polygon_y = [y for index, y in enumerate(ann_seg) if index % 2 == 1]
    polygon_xy = [[x, y] for x, y in zip(polygon_x, polygon_y)]
    polygon_xy = np.array(polygon_xy, np.int32)
    return polygon_xy

def get_mask(image_array_shape, polygon_xy):
    mask = np.zeros(image_array_shape)
    masked_polygon = cv2.fillPoly(mask, [polygon_xy], 1)
    
    return masked_polygon

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

# ann_seg_list에 있는 object들의 segmentation에 따라 instance segmentation 시각화. 
def draw_segment(image_array, ann_seg_list, color_list, alpha):
    
    draw_image = image_array.copy()
    mask_array_shape = draw_image.shape[0:2]
    
    # list형태로 입력된 segmentation 정보들을 각각 시각화
    for index, ann_seg in enumerate(ann_seg_list):
        # polygon 좌표로 변환. 
        polygon_xy = get_polygon_xy(ann_seg)
        # mask 정보 변환
        masked_polygon = get_mask(mask_array_shape, polygon_xy)
        
        # segmentation color와 외곽선용 color 선택
        color_object = color_list[np.random.randint(len(color_list))]
        color_contour = color_list[np.random.randint(len(color_list))]
        # masking 적용
        masked_image = apply_mask(draw_image, masked_polygon, color_object, alpha=0.6)
        masked_image_copy = masked_image.copy()
        # 외곽선 적용. 
        s_mask_int = (masked_polygon * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(s_mask_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    
    return cv2.drawContours(masked_image_copy, contours, -1, color_contour, 1, cv2.LINE_8, hierarchy, 100)

def get_coco_masked_image(coco, image_id, image_name):
    annIds = coco.getAnnIds(imgIds=[image_id], catIds=[3], iscrowd=None)
    anns = coco.loadAnns(annIds)
    # segmentation 정보만 별도로 추출. 
    ann_seg_list = [ann['segmentation'] for ann in anns]
    
    image_array = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
    # coco segmentation 정보를 기반으로 segmentation 적용한 이미지 시각화 
    masked_image = draw_segment(image_array, ann_seg_list, color_list, alpha=0.6)
    
    return masked_image

def show_coco_masked_image(coco, data_df, image_id_list, cols=5):
    
    figure, axs = plt.subplots(nrows=1, ncols=cols, figsize=(16, 12))
    for i in range(cols):
        image_id = image_id_list[i]
        image_name = data_df[data_df['image_id'] == image_id]['image_name'].to_list()[0]
        masked_image = get_coco_masked_image(coco, image_id, image_name)
        
        axs[i].imshow(masked_image)
        
def show_coco_image(data_df, image_id_list, cols=5):
    
    figure, axs = plt.subplots(nrows=1, ncols=cols, figsize=(16, 12))
    for i in range(cols):
        image_id = image_id_list[i]
        image_name = data_df[data_df['image_id'] == int(image_id)]['image_name'].to_list()[0]

        axs[i].imshow(cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB))

@DATASETS.register_module(force=True)
class BloomDataset(CocoDataset):
    CLASSES = ['Crack',
               'billet',
               'Shrinkage',
               'Equiaxed_Area',
               'Scratch',
               'NotQuenching_Profile',
               'Pinhole',
               'Bleeding',
               'Bloom',
               'Bloom_TB_Surface',
               'Bloom_Side_Surface',
               'L_direction']

# 포함 여부에 따라 kind 값을 정해주는 함수
def assign_kind(meta_df):
    """
    meta_df 내 kind 열을 새로 생성하고
    meta_df['maskname_list'] 내 원소가 billet가 포함되면 0,
    Bloom가 포함되면 1,
    Bloom_TB_Surface가 포함되면 2,
    Bloom_Side_Surface가 포함되면 3,
    L_direction가 포함되면 4,
    Bloom과 NotQuenching_Profile이 포함되면 5,
    Bloom_TB_Surface 또는 Bloom_Side_Surface이면서 Scratch이 포함되면 6,
    Bloom_TB_Surface 또는 Bloom_Side_Surface이면서 Bleeding이 포함되면 7로 만드는 함수
    
    Args:
    - meta_df : DataFrame 형태의 메타 데이터. maskname_list 열이 있어야 함.
    
    Returns:
    - kind 열이 추가된 meta_df
    
    """
    maskname_list = meta_df['maskname_list'].values
    kind_list = np.zeros(len(maskname_list), dtype=int)
    for i, tags in enumerate(maskname_list):
        if not tags:
            continue
        tags = sum(tags, [])  # 이중 리스트를 하나의 리스트로 펼침

        if any(tag for tag in tags if 'NotQuenching_Profile' in tag):
            kind_list[i] = 5
        elif any(tag for tag in tags if 'Scratch' in tag):
            kind_list[i] = 6
        elif any(tag for tag in tags if 'Bleeding' in tag):
            kind_list[i] = 7
        elif any(tag for tag in tags if 'Bloom_TB_Surface' in tag):
            kind_list[i] = 2
        elif any(tag for tag in tags if 'Bloom_Side_Surface' in tag):
            kind_list[i] = 3
        elif any(tag for tag in tags if 'Bloom' in tag):
            kind_list[i] = 1
        elif any(tag for tag in tags if 'L_direction' in tag):
            kind_list[i] = 4
        elif any(tag for tag in tags if 'Shrinkage' in tag):
            kind_list[i] = 9      
        elif any(tag for tag in tags if 'Equiaxed_Area' in tag):
            kind_list[i] = 10  
        elif any(tag for tag in tags if 'Crack' in tag):
            kind_list[i] = 8 
        elif any(tag for tag in tags if 'billet' in tag):
            kind_list[i] = 0
            
    meta_df['kind'] = kind_list
    return meta_df

if __name__ == '__main__':
    labels_to_names_seq =  {0:'Crack', 
                            1:'billet', 
                            2:'Shrinkage', 
                            3:'Equiaxed_Area',
                            4:'Scratch',
                            5:'NotQuenching_Profile',
                            6:'Pinhole',
                            7:'Bleeding',
                            8:'Bloom',
                            9:'Bloom_TB_Surface',
                            10:'Bloom_Side_Surface',
                            11:'L_direction'}

    DATA_ROOT_DIR = 'bloom'
    color_list = [
                (0, 255, 0),
                (255, 0, 0),
                (0, 0, 255)
    ]
    
    # 이미지와 마스크 경로집합인 meta dataframe 생성
    print('################ 이미지와 마스크 경로집합인 meta dataframe 생성 ######################')
    meta_df = make_meta_df(DATA_ROOT_DIR)
    
    # 샘플종류별 데이터 분포 밸런스를 맞추기 위해 kind 열을 추가
    assign_meta_df = assign_kind(meta_df)
    assign_meta_df.to_excel('coco_bloom/assign_meta_df.xlsx')
    print(assign_meta_df['kind'].unique())

    # 학습과 검증용 image id 데이터들을 추출. 
    print('################ 학습과 검증용 image id 데이터들을 추출 ######################')
    
    # 1. 샘플분류모델개발 : train 248 / val 48 
    # train_df, val_df = train_test_split(assign_meta_df, test_size=0.2, random_state=2023, stratify=meta_df['kind'])
    # val_df, test_df = train_test_split(val_df, test_size=0.8, random_state=2022)
    # train_df, val_df = train_test_split(assign_meta_df, test_size=0.7, random_state=2023, stratify=meta_df['kind'])
    # val_df, test_df = train_test_split(val_df, test_size=0.8, random_state=2022)
    # 2. 빌렛단면 크랙모델개발 : train 88 / val 23 
    # train_df, val_df = train_test_split(assign_meta_df, test_size=0.2, random_state=2023)
    # 3. 블룸단면 크랙모델개발 : train 30 / val 18
    # train_df, val_df = train_test_split(meta_df, test_size=0.6, random_state=2023, stratify=meta_df['kind'])
    # val_df, test_df = train_test_split(val_df, test_size=0.6, random_state=2023)
    # 4. 블룸L방향 크랙모델개발 : train 20 / val 6
    # train_df, val_df = train_test_split(assign_meta_df, test_size=0.2, random_state=2023, stratify=meta_df['kind'])
    # 5. 블룸표면 크랙모델개발 : train 12 1016 / val 6 666
    # train_df, val_df = train_test_split(assign_meta_df, test_size=0.2, random_state=2023, stratify=meta_df['kind'])
    train_df, val_df = train_test_split(assign_meta_df, test_size=0.88, random_state=800, stratify=meta_df['kind'])
    val_df, test_df = train_test_split(val_df, test_size=0.96, random_state=800)
    
    train_df.to_excel('coco_bloom/train_df.xlsx')
    val_df.to_excel('coco_bloom/val_df.xlsx')
    print('train_test_split 순서 저장완료')
    train_ids = train_df['image_id'].to_list()
    val_ids = val_df['image_id'].to_list()
    print('################ meta_df 분포 밸런스 검토 ######################')
    print(meta_df['kind'].value_counts().sort_index())
    print('################ 학습용 데이터 분포 밸런스 검토 ######################')
    print(train_df['kind'].value_counts().sort_index())
    print('################ 검증용 데이터 분포 밸런스 검토 ######################')
    print(val_df['kind'].value_counts().sort_index())
    
    # COCO json 파일 생성.
    print('################ COCO json 파일 생성 ######################')
    convert_bloom_to_coco('bloom', train_ids, 'coco_bloom/train_coco.json', 'coco_bloom/train', meta_df, labels_to_names_seq)
    convert_bloom_to_coco('bloom', val_ids, 'coco_bloom/val_coco.json', 'coco_bloom/val', meta_df, labels_to_names_seq)
    print('################ Crack 이진분류 COCO json 파일 생성 ######################')

    coco_train = COCO('coco_bloom/train_coco.json')
    coco_val= COCO('coco_bloom/val_coco.json')
    
    # config 파일을 설정하고, 다운로드 받은 pretrained 모델을 checkpoint로 설정.
    # config_file = 'mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
    # checkpoint_file = 'mmdetection/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
    config_file = 'mmdetection/configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco.py'
    checkpoint_file = 'tutorial_exps/bloom_segment_model/블룸표면/epoch_36.pth'
    
    cfg = Config.fromfile(config_file)
    
    ###################################################Config조정#############################################################
    # 이미지해상도 조정 : 안함
    # defalut img_scale = (1333, 800) # 샘플검출 모델학습
    # img_scale = (1333, 800) # 샘플검출 모델학습
    # img_scale = (333, 200)
    # img_scale = (500, 300)
    img_scale = (2576, 1932) # 2k 크랙검출 모델 학습
    # img_scale = (2060, 1546) # 기본해상도의 1.8배로 크랙검출 모델 학습
    cfg.train_pipeline[2]['img_scale'] = img_scale
    cfg.test_pipeline[1]['img_scale'] = img_scale #img_scale
    cfg.data.train.pipeline[2]['img_scale'] = img_scale
    cfg.data.train.pipeline[2]['img_scale'] = img_scale
    cfg.data.val.pipeline[1]['img_scale'] = img_scale #img_scale
    cfg.data.test.pipeline[1]['img_scale'] = img_scale #img_scale

    # batch size = 2로 변경
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=16)

    #----------------------------------------------------------------
    print(
    cfg.train_pipeline[2]['img_scale'],
    # cfg.train_pipeline[3]['img_scale'],
    cfg.test_pipeline[1]['img_scale'],
    cfg.data.train.pipeline[2]['img_scale'],
    # cfg.data.train.pipeline[3]['img_scale'],
    # cfg.data.train.pipeline[2]['img_scale'],
    cfg.data.val.pipeline[1]['img_scale'],
    cfg.data.test.pipeline[1]['img_scale'],
    cfg.auto_scale_lr)
    #----------------------------------------------------------------

    # dataset에 대한 환경 파라미터 수정. 
    cfg.dataset_type = 'BloomDataset'
    # cfg.data.train.dataset.type  = 'BilletDataset'
    # cfg.dataset.type = 'BilletDataset'
    cfg.data_root = 'coco_bloom/'

    # del cfg['data']['train']['times']

    # train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
    cfg.data.train.type = 'BloomDataset'
    cfg.data.train.data_root = 'coco_bloom/'
    cfg.data.train.ann_file = 'train_coco.json'
    cfg.data.train.img_prefix = 'train'

    # cfg.data.train.dataset.ann_file = 'train_coco.json'
    # cfg.data.train.dataset.img_prefix = 'train'

    # cfg.data.train.seg_prefix = ''

    cfg.data.val.type = 'BloomDataset'
    cfg.data.val.data_root = 'coco_bloom/'
    cfg.data.val.ann_file = 'val_coco.json'
    cfg.data.val.img_prefix = 'val'
    # cfg.data.val.seg_prefix = ''
    # cfg.data.val.ins_ann_file = None

    cfg.data.test.type = 'BloomDataset'
    cfg.data.test.data_root = 'coco_bloom/'
    cfg.data.test.ann_file = 'val_coco.json'
    cfg.data.test.img_prefix = 'val'
    # cfg.data.test.seg_prefix = ''
    # cfg.data.test.ins_ann_file = None

    # # class의 갯수 수정. 
    # cfg.model.roi_head.bbox_head.num_classes = 1
    # cfg.model.roi_head.mask_head.num_classes = 1
    
    cfg.model.roi_head.bbox_head.num_classes = 12
    cfg.model.roi_head.mask_head.num_classes = 12
    # mask2transformer의 num_classes
    # cfg.num_things_classes  =  4
    # cfg.num_stuff_classes  =  0
    # cfg.num_classes  =  cfg.num_things_classes  +  cfg.num_stuff_classes

    # pretrained 모델
    cfg.load_from = checkpoint_file
    
    # 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정. 

    cfg.work_dir = 'tutorial_exps/bloom_segment_model'

    # 학습율 변경 환경 파라미터 설정. 
    cfg.optimizer.lr = 0.02 / 8
    # cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # CocoDataset의 경우 metric을 bbox로 설정해야 함.(mAP아님. bbox로 설정하면 mAP를 iou threshold를 0.5 ~ 0.95까지 변경하면서 측정)
    cfg.evaluation.metric = ['bbox', 'segm']
    cfg.evaluation.interval = 36
    cfg.checkpoint_config.interval = 36

    # epochs 횟수는 5000으로 증가 
    cfg.runner.max_epochs = 10000
    # cfg.interval = 10
    # cfg.runner.max_iters = 36
    # cfg.max_iters = 36
    # 두번 config를 로드하면 lr_config의 policy가 사라지는 오류로 인하여 설정. 
    cfg.lr_config.policy = 'step'
    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # ConfigDict' object has no attribute 'device 오류 발생시 반드시 설정 필요. https://github.com/open-mmlab/mmdetection/issues/7901
    cfg.device='cuda'
    cfg.model.backbone.with_cp = True
    # TEST시 max_per_image= 100에서 25 감소시키면 CUDA OOM문제 감소가능
    # cfg.model.test_cfg.max_per_image = 25
    ###################################################Config조정#############################################################
    
    # train, valid 용 Dataset 생성. 
    datasets_train = [build_dataset(cfg.data.train)]
    datasets_val = [build_dataset(cfg.data.val)]
    
    print(datasets_train)
    print(datasets_val)
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets_train[0].CLASSES
    print(model.CLASSES)
    
    # mask2former swin small model 학습시작
    print('################ mask_rcnn_x101_64x4d_fpn_1x_coco model 학습시작 ######################')
    import os.path as osp
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # epochs는 config의 runner 파라미터로 지정됨. 기본 12회 
    from mmdet.utils import AvoidCUDAOOM

    torch.cuda.empty_cache()
    learn = train_detector(model, datasets_train, cfg, distributed=False, validate=True)
    # output = AvoidCUDAOOM.retry_if_cuda_oom(train_detector(model, datasets_train, cfg, distributed=False, validate=True))(datasets_train, datasets_val)
