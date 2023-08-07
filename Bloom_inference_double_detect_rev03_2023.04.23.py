import warnings
warnings.filterwarnings('ignore')
import time
start_time = time.time()
from mmdet.apis import init_detector, inference_detector
import os
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', 300)
import matplotlib.pyplot as plt
import cv2
import json
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmcv import Config
from mmdet.apis import set_random_seed
import seaborn as sns
import sys
import math

# print(sys.argv)
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

def evaluate_mAP(file_path):
    json_data = []
    train = []
    val = []
    dataframe = pd.DataFrame()
    with open(file_path, 'r') as f:
        for line in f:
            # print(json.loads(line))
            json_data.append(json.loads(line))
            for i in range(len(json_data)):
                if json_data[i]['mode']=='train':
                    train.append(json_data[i])
                    
                else:
                    val.append(json_data[i])
                    
    train = pd.DataFrame(train)
    val = pd.DataFrame(val)
    return train, val

def visualization(val, ymin=0, ymax=0.45, xmin=0, xmax=36):
    # 1. bbox vs segmentation mAP with epoch plot
    plt.figure(figsize=(10, 6))
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    sns.lineplot(x=val.epoch, y=val.segm_mAP, data=val, label='segmentation_mAP')
    sns.scatterplot(x=val.epoch, y=val.segm_mAP, data=val)
    sns.lineplot(x=val.epoch, y=val.bbox_mAP, data=val, label='bounding_box_mAP')
    sns.scatterplot(x=val.epoch, y=val.bbox_mAP, data=val)
    plt.title('segm_mAP with epoch')
    plt.show()

    # 2. segmentation small, middle, large mAP with epoch plot
    plt.figure(figsize=(10, 6))
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    sns.lineplot(x=val.epoch, y=val.segm_mAP_s, data=val, label='segmentation_small_object_mAP')
    sns.scatterplot(x=val.epoch, y=val.segm_mAP_s, data=val)
    sns.lineplot(x=val.epoch, y=val.segm_mAP_m, data=val, label='segmentation_middle_object_mAP')
    sns.scatterplot(x=val.epoch, y=val.segm_mAP_m, data=val)
    sns.lineplot(x=val.epoch, y=val.segm_mAP_l, data=val, label='segmentation_large_object_mAP')
    sns.scatterplot(x=val.epoch, y=val.segm_mAP_l, data=val)
    plt.title('segmentation small, middle, large mAP')
    plt.show()

    # 3. bounding box small, middle, large mAP with epoch plot
    plt.figure(figsize=(10, 6))
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    sns.lineplot(x=val.epoch, y=val.bbox_mAP_s, data=val, label='bounding_box_small_object_mAP')
    sns.scatterplot(x=val.epoch, y=val.bbox_mAP_s, data=val)
    sns.lineplot(x=val.epoch, y=val.bbox_mAP_m, data=val, label='bounding_box_middle_object_mAP')
    sns.scatterplot(x=val.epoch, y=val.bbox_mAP_m, data=val)
    sns.lineplot(x=val.epoch, y=val.bbox_mAP_l, data=val, label='bounding_box_large_object_mAP')
    sns.scatterplot(x=val.epoch, y=val.bbox_mAP_l, data=val)
    plt.title('bounding box small, middle, large mAP')
    plt.show()
    
def bloom_cross_section_crack_classification(df, MACRO, p_width, p_height):
    data = df.copy()

    p1=p_width[0] 
    p2=p_width[1] 
    p3=p_width[2] # p8
    p4=p_width[3] 
    p5=p_width[4] # p9
    p6=p_width[5] 
    p7=p_width[6] 
    p8=p_height[2]
    p9=p_height[4]
    
    
    #### 1) 특정샘플의 좌표를 정의한다
    bilet_left = data[(data['class'] == f'{MACRO}')]['left'].tolist()[0]
    bilet_top = data[(data['class'] ==f'{MACRO}')]['top'].tolist()[0]
    bilet_right = data[(data['class'] ==f'{MACRO}')]['right'].tolist()[0]
    bilet_bottom = data[(data['class'] ==f'{MACRO}')]['bottom'].tolist()[0]
    
    #### 2) 샘플좌표 대비 Object 가로세로 비율을 정의한다
    data['left_ratio'] = abs(data['left'] - bilet_left) / (bilet_right - bilet_left) * 100
    data['right_ratio'] = abs(data['right'] - bilet_left) / (bilet_right - bilet_left) * 100
    data['top_ratio'] = abs(data['top'] - bilet_top) / (bilet_bottom - bilet_top) * 100
    data['bottom_ratio'] = abs(data['bottom'] - bilet_top) / (bilet_bottom - bilet_top) * 100
    
    
    data['centerX_ratio'] = (data['left_ratio'] + data['right_ratio']) / 2
    data['centerY_ratio'] = (data['top_ratio'] + data['bottom_ratio']) / 2
    
    #### 3) Object 가로세로 비율을 기준으로 영역번호를 부여한다
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  <= p1), 'Position_number'] = 1
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 7
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p8), 'Position_number'] = 7
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  > p3) & (data['centerY_ratio']  <= p4), 'Position_number'] = 21
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p9), 'Position_number'] = 29
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  > p5) & (data['centerY_ratio']  <= p6), 'Position_number'] = 37
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 37
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 51
    
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  <= p1), 'Position_number'] = 2
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 8
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p8), 'Position_number'] = 15
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  > p8) & (data['centerY_ratio']  <= p4), 'Position_number'] = 22
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p9), 'Position_number'] = 30
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  > p9) & (data['centerY_ratio']  <= p6), 'Position_number'] = 38
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 45
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 52

    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  <= p1), 'Position_number'] = 2
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 9
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p8), 'Position_number'] = 16
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  > p8) & (data['centerY_ratio']  <= p4), 'Position_number'] = 23
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p9), 'Position_number'] = 31
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  > p9) & (data['centerY_ratio']  <= p6), 'Position_number'] = 39
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 46
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 52

    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  <= p1), 'Position_number'] = 3
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 10
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p8), 'Position_number'] = 17
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  > p8) & (data['centerY_ratio']  <= p4), 'Position_number'] = 24
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p9), 'Position_number'] = 32
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  > p9) & (data['centerY_ratio']  <= p6), 'Position_number'] = 40
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 47
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 53

    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  <= p1), 'Position_number'] = 4
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 11
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p8), 'Position_number'] = 18
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  > p8) & (data['centerY_ratio']  <= p4), 'Position_number'] = 25
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p9), 'Position_number'] = 33
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  > p9) & (data['centerY_ratio']  <= p6), 'Position_number'] = 41
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 48
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 54

    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  <= p1), 'Position_number'] = 5
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 12
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p8), 'Position_number'] = 19
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  > p8) & (data['centerY_ratio']  <= p4), 'Position_number'] = 26
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p9), 'Position_number'] = 34
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  > p9) & (data['centerY_ratio']  <= p6), 'Position_number'] = 42
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 49
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 55
    
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  <= p1), 'Position_number'] = 5
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 13
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p8), 'Position_number'] = 20
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  > p8) & (data['centerY_ratio']  <= p4), 'Position_number'] = 27
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p9), 'Position_number'] = 35
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  > p9) & (data['centerY_ratio']  <= p6), 'Position_number'] = 43
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 50
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 55
    
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  <= p1), 'Position_number'] = 6
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 14
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p8), 'Position_number'] = 14
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  > p8) & (data['centerY_ratio']  <= p4), 'Position_number'] = 28
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p9), 'Position_number'] = 36
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  > p9) & (data['centerY_ratio']  <= p6), 'Position_number'] = 44
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 44
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 56
    
    # 샘플은 (width_ratio, height_ratio) = (0,0)으로 0으로 무시한다
    data.loc[(data['centerX_ratio'] < 1) & (data['centerY_ratio']  < 1), 'Position_number'] = 0
    
    #### 4) 위치column 생성하고 영역번호 기준으로 위치과 결함을 분류하는 rule-base를 정의한다
    ############### 위치 정의 ###############
    conditions = [
        (data['Position_number'].isin([1, 8, 16, 2, 9])),  
        (data['Position_number'].isin([5, 6, 12, 13, 19])), 
        (data['Position_number'].isin([45, 46, 39, 52, 51])), 
        (data['Position_number'].isin([42, 49, 50, 55, 56])), 
        (data['Position_number'].isin([7, 15])), 
        (data['Position_number'].isin([38, 37])), 
        (data['Position_number'].isin([20, 14])), 
        (data['Position_number'].isin([44, 43])),
        
        (data['Position_number'].isin([3,4,10,11,17,18,24,25])),
        (data['Position_number'].isin([32,33,40,41,47,48,53,54])),
        (data['Position_number'].isin([21,22,23,29,30,31])),
        (data['Position_number'].isin([26,27,28,34,35,36])),
    ]
    choices = ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'left_top', 'left_bottom', 'right_top', 'right_bottom', 'top', 'bottom', 'left', 'right']
    
    ############### 결함 정의 ###############
    # (1) 대각크랙
    diagonal = [8,16,13,19,45,39,50,42]
    # (2) 힌지크랙
    hinge = [2,5,52,55,7,37,14,44, 43, 49, 38, 15, 9, 12, 20]
    # (3) 미드웨이크랙1
    midway1 = [3,4,53,54,21,29,28,36]
    # (4) 표면직하크랙
    subsurface_crack = [1,6,51,56]
    # (5) 미드웨이크랙2
    midway2 = [10,17,11,18,40,46,47,41,18,22,23,30,31,26,27,34,35]
    # (6) 중심부 SR크랙
    center_SR = [24,25,32,33]
    # (7) 샘플과 등축정은 제외한다
    sample = [0]
    # dictionary로 저장
    crack_dict = {
        'diagonal': diagonal,
        'hinge': hinge,
        'midway1': midway1,
        'subsurface_crack': subsurface_crack,
        'midway2': midway2,
        'center_SR': center_SR,
        'sample': sample
    }

    # loop을 사용하여 class_kind, Position 열 값을 설정
    for key, value in crack_dict.items():
        data.loc[data['Position_number'].isin(value), 'Position'] = key
        data.loc[data['Position_number'].isin(value), 'class_kind'] = key
        
    # Equiaxed_Area의 class도 'Sample'로 변경
    data.loc[data['class'] == 'Equiaxed_Area', ['class_kind', 'Position']] = 'sample'
    data['Position'] = np.select(conditions, choices)
    data.loc[
        (data['Position_number'] == 0), 'class_kind'] = 'Sample'
    data.loc[
        (data['Position_number'] == 0), 'Position'] = 'Sample'
    data.loc[
        (data['class'] == 'Equiaxed_Area'), 'class_kind'] = 'Sample'
    data.loc[
        (data['class'] == 'Equiaxed_Area'), 'Position'] = 'Sample'
    data.loc[
        (data['class'] == 'Crack') & (data['class_kind'] == 'diagonal') & (data['bbox_angle'] <= 40) | (data['class'] == 'Crack') & (data['class_kind'] == 'diagonal') & (data['bbox_angle'] >= 70), 'class_kind'] = 'hinge'
    
    
    #### 5) DataFrame으로 추가한다
    if (data.columns[-1:].to_list()[0] == 'class_kind') | (data.columns[-1:].to_list()[0] == 'Position'):        
        col1 = data.columns[:2].to_list()
        col2 = data.columns[-3:].to_list()
        col3 = data.columns[2:-3].to_list()
        new_col = col1 + col2 + col3
        data = data[new_col]
    else:
        pass
    
    return data

def billet_cross_section_crack_classification(df, MACRO, p1=8.3, p2=20.8, p3=33.3, p4=50, p5=66.7, p6=79.2, p7=91.7):
    data = df.copy()

    #### 1) 특정샘플의 좌표를 정의한다
    bilet_left = data[(data['class'] == f'{MACRO}')]['left'].tolist()[0]
    bilet_top = data[(data['class'] ==f'{MACRO}')]['top'].tolist()[0]
    bilet_right = data[(data['class'] ==f'{MACRO}')]['right'].tolist()[0]
    bilet_bottom = data[(data['class'] ==f'{MACRO}')]['bottom'].tolist()[0]
    
    #### 2) 샘플좌표 대비 Object 가로세로 비율을 정의한다
    data['left_ratio'] = abs(data['left'] - bilet_left) / (bilet_right - bilet_left) * 100
    data['right_ratio'] = abs(data['right'] - bilet_left) / (bilet_right - bilet_left) * 100
    data['top_ratio'] = abs(data['top'] - bilet_top) / (bilet_bottom - bilet_top) * 100
    data['bottom_ratio'] = abs(data['bottom'] - bilet_top) / (bilet_bottom - bilet_top) * 100
    
    
    data['centerX_ratio'] = (data['left_ratio'] + data['right_ratio']) / 2
    data['centerY_ratio'] = (data['top_ratio'] + data['bottom_ratio']) / 2
    
    #### 3) Object 가로세로 비율을 기준으로 영역번호를 부여한다
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  <= p1), 'Position_number'] = 1
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 7
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p3), 'Position_number'] = 7
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  > p3) & (data['centerY_ratio']  <= p4), 'Position_number'] = 21
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p5), 'Position_number'] = 29
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  > p5) & (data['centerY_ratio']  <= p6), 'Position_number'] = 37
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 37
    data.loc[(data['centerX_ratio'] <= p1) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 51
    
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  <= p1), 'Position_number'] = 2
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 8
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p3), 'Position_number'] = 15
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  > p3) & (data['centerY_ratio']  <= p4), 'Position_number'] = 22
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p5), 'Position_number'] = 30
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  > p5) & (data['centerY_ratio']  <= p6), 'Position_number'] = 38
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 45
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 52

    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  <= p1), 'Position_number'] = 2
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 9
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p3), 'Position_number'] = 16
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  > p3) & (data['centerY_ratio']  <= p4), 'Position_number'] = 23
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p5), 'Position_number'] = 31
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  > p5) & (data['centerY_ratio']  <= p6), 'Position_number'] = 39
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 46
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 52

    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  <= p1), 'Position_number'] = 3
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 10
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p3), 'Position_number'] = 17
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  > p3) & (data['centerY_ratio']  <= p4), 'Position_number'] = 24
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p5), 'Position_number'] = 32
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  > p5) & (data['centerY_ratio']  <= p6), 'Position_number'] = 40
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 47
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 53

    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  <= p1), 'Position_number'] = 4
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 11
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p3), 'Position_number'] = 18
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  > p3) & (data['centerY_ratio']  <= p4), 'Position_number'] = 25
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p5), 'Position_number'] = 33
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  > p5) & (data['centerY_ratio']  <= p6), 'Position_number'] = 41
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 48
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 54

    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  <= p1), 'Position_number'] = 5
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 12
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p3), 'Position_number'] = 19
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  > p3) & (data['centerY_ratio']  <= p4), 'Position_number'] = 26
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p5), 'Position_number'] = 34
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  > p5) & (data['centerY_ratio']  <= p6), 'Position_number'] = 42
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 49
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 55
    
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  <= p1), 'Position_number'] = 5
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 13
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p3), 'Position_number'] = 20
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  > p3) & (data['centerY_ratio']  <= p4), 'Position_number'] = 27
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p5), 'Position_number'] = 35
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  > p5) & (data['centerY_ratio']  <= p6), 'Position_number'] = 43
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 50
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 55
    
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  <= p1), 'Position_number'] = 6
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  > p1) & (data['centerY_ratio']  <= p2), 'Position_number'] = 14
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  > p2) & (data['centerY_ratio']  <= p3), 'Position_number'] = 14
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  > p3) & (data['centerY_ratio']  <= p4), 'Position_number'] = 28
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  > p4) & (data['centerY_ratio']  <= p5), 'Position_number'] = 36
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  > p5) & (data['centerY_ratio']  <= p6), 'Position_number'] = 44
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  > p6) & (data['centerY_ratio']  <= p7), 'Position_number'] = 44
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100) & (data['centerY_ratio']  > p7) & (data['centerY_ratio']  <= 100), 'Position_number'] = 56
    
    # 샘플은 (width_ratio, height_ratio) = (0,0)으로 0으로 무시한다
    data.loc[(data['centerX_ratio'] < 1) & (data['centerY_ratio']  < 1), 'Position_number'] = 0
    
    #### 4) 위치column 생성하고 영역번호 기준으로 위치과 결함을 분류하는 rule-base를 정의한다
    ############### 위치 정의 ###############
    conditions = [
        (data['Position_number'].isin([1, 8, 16, 2, 9])),  
        (data['Position_number'].isin([5, 6, 12, 13, 19])), 
        (data['Position_number'].isin([45, 46, 39, 52, 51])), 
        (data['Position_number'].isin([42, 49, 50, 55, 56])), 
        (data['Position_number'].isin([7, 15])), 
        (data['Position_number'].isin([38, 37])), 
        (data['Position_number'].isin([20, 14])), 
        (data['Position_number'].isin([44, 43])),
        
        (data['Position_number'].isin([3,4,10,11,17,18,24,25])),
        (data['Position_number'].isin([32,33,40,41,47,48,53,54])),
        (data['Position_number'].isin([21,22,23,29,30,31])),
        (data['Position_number'].isin([26,27,28,34,35,36])),
    ]
    choices = ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'left_top', 'left_bottom', 'right_top', 'right_bottom', 'top', 'bottom', 'left', 'right']
    
    ############### 결함 정의 ###############
    # (1) 대각크랙
    diagonal = [8,16,13,19,45,39,50,42]
    # (2) 힌지크랙
    hinge = [2,5,52,55,7,37,14,44]
    # (3) 미드웨이크랙1
    midway1 = [3,4,53,54,21,29,28,36]
    # (4) 표면직하크랙
    subsurface_crack = [1,6,51,56]
    # (5) 미드웨이크랙2
    midway2 = [9,10,17,11,12,18,40,46,47,41,18,49,15,22,23,30,31,38,20,26,27,34,35,43]
    # (6) 중심부 SR크랙
    center_SR = [24,25,32,33]
    # (7) 샘플과 등축정은 제외한다
    sample = [0]
    # dictionary로 저장
    crack_dict = {
        'diagonal': diagonal,
        'hinge': hinge,
        'midway1': midway1,
        'subsurface_crack': subsurface_crack,
        'midway2': midway2,
        'center_SR': center_SR,
        'sample': sample
    }

    # loop을 사용하여 class_kind, Position 열 값을 설정
    for key, value in crack_dict.items():
        data.loc[data['Position_number'].isin(value), 'Position'] = key
        data.loc[data['Position_number'].isin(value), 'class_kind'] = key
        
    # Equiaxed_Area의 class도 'Sample'로 변경
    data.loc[data['class'] == 'Equiaxed_Area', ['class_kind', 'Position']] = 'sample'
    data['Position'] = np.select(conditions, choices)
    data.loc[
        (data['Position_number'] == 0), 'class_kind'] = 'Sample'
    data.loc[
        (data['Position_number'] == 0), 'Position'] = 'Sample'
    data.loc[
        (data['class'] == 'Equiaxed_Area'), 'class_kind'] = 'Sample'
    data.loc[
        (data['class'] == 'Equiaxed_Area'), 'Position'] = 'Sample'
    data.loc[
        (data['class'] == 'Crack') & (data['class_kind'] == 'diagonal') & (data['bbox_angle'] <= 40) | (data['class'] == 'Crack') & (data['class_kind'] == 'diagonal') & (data['bbox_angle'] >= 60), 'class_kind'] = 'hinge'
    
    
    #### 5) DataFrame으로 추가한다
    if (data.columns[-1:].to_list()[0] == 'class_kind') | (data.columns[-1:].to_list()[0] == 'Position'):        
        col1 = data.columns[:2].to_list()
        col2 = data.columns[-3:].to_list()
        col3 = data.columns[2:-3].to_list()
        new_col = col1 + col2 + col3
        data = data[new_col]
    else:
        pass
    
    return data    
   
def bloom_surface_crack_classification(df, MACRO, p1=8.3, p2=20.8, p3=33.3, p4=50, p5=66.7, p6=79.2, p7=91.7):
    data = df.copy()

    #### 1) 특정샘플의 좌표를 정의한다
    bilet_left = data[(data['class'] == f'{MACRO}')]['left'].tolist()[0]
    bilet_top = data[(data['class'] ==f'{MACRO}')]['top'].tolist()[0]
    bilet_right = data[(data['class'] ==f'{MACRO}')]['right'].tolist()[0]
    bilet_bottom = data[(data['class'] ==f'{MACRO}')]['bottom'].tolist()[0]
    
    #### 2) 샘플좌표 대비 Object 가로세로 비율을 정의한다
    data['left_ratio'] = abs(data['left'] - bilet_left) / (bilet_right - bilet_left) * 100
    data['right_ratio'] = abs(data['right'] - bilet_left) / (bilet_right - bilet_left) * 100
    data['top_ratio'] = abs(data['top'] - bilet_top) / (bilet_bottom - bilet_top) * 100
    data['bottom_ratio'] = abs(data['bottom'] - bilet_top) / (bilet_bottom - bilet_top) * 100
    
    
    data['centerX_ratio'] = (data['left_ratio'] + data['right_ratio']) / 2
    data['centerY_ratio'] = (data['top_ratio'] + data['bottom_ratio']) / 2
    
    #### 3) Object 가로 비율을 기준으로 영역번호를 부여한다
    data.loc[(data['centerX_ratio'] <= p1), 'Position_number'] = 1
    data.loc[(data['centerX_ratio'] > p1) & (data['centerX_ratio'] <= p2), 'Position_number'] = 2
    data.loc[(data['centerX_ratio'] > p2) & (data['centerX_ratio'] <= p3), 'Position_number'] = 3
    data.loc[(data['centerX_ratio'] > p3) & (data['centerX_ratio'] <= p4), 'Position_number'] = 4
    data.loc[(data['centerX_ratio'] > p4) & (data['centerX_ratio'] <= p5), 'Position_number'] = 5
    data.loc[(data['centerX_ratio'] > p5) & (data['centerX_ratio'] <= p6), 'Position_number'] = 6
    data.loc[(data['centerX_ratio'] > p6) & (data['centerX_ratio'] <= p7), 'Position_number'] = 7
    data.loc[(data['centerX_ratio'] > p7) & (data['centerX_ratio'] <= 100), 'Position_number'] = 8

    
    # 샘플은 (width_ratio, height_ratio) = (0,0)으로 0으로 무시한다
    data.loc[(data['centerX_ratio'] < 1) & (data['centerY_ratio']  < 1), 'Position_number'] = 0
    
    #### 4) 위치column 생성하고 영역번호 기준으로 위치과 결함을 분류하는 rule-base를 정의한다
    ############### 위치 정의 ###############
    
    if MACRO == 'Bloom_TB_Surface':
        conditions = [
            (data['Position_number'].isin([1])),  
            (data['Position_number'].isin([2])), 
            (data['Position_number'].isin([3])), 
            (data['Position_number'].isin([4,5])), 
            (data['Position_number'].isin([6])), 
            (data['Position_number'].isin([7])), 
            (data['Position_number'].isin([8]))]
        choices = ['left_corner', 'left_nearcorner', 'left_face', 'face', 'right_face', 'right_nearcorner', 'right_corner']
    elif (MACRO == 'Bloom_Side_Surface') | (MACRO == 'L_direction'):
        conditions = [
            (data['Position_number'].isin([1])),  
            (data['Position_number'].isin([2])), 
            (data['Position_number'].isin([3])), 
            (data['Position_number'].isin([4,5])), 
            (data['Position_number'].isin([6])), 
            (data['Position_number'].isin([7])), 
            (data['Position_number'].isin([8]))]
        choices = ['top_corner', 'top_nearcorner', 'top_face', 'face', 'bottom_face', 'bottom_nearcorner', 'bottom_corner']
    ############### 결함 정의 ###############
    # (1) 의미없는 값으로 에러나지 않기 위해 데이터프레임 열을 채우기 위한 코드
    sample = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # dictionary로 저장
    crack_dict = {
        'sample': sample
    }

    # loop을 사용하여 class_kind, Position 열 값을 설정
    for key, value in crack_dict.items():
        data.loc[data['Position_number'].isin(value), 'Position'] = key
        data.loc[data['Position_number'].isin(value), 'class_kind'] = key
        
    # Equiaxed_Area의 class도 'Sample'로 변경
    data.loc[data['class'] == 'Equiaxed_Area', ['class_kind', 'Position']] = 'sample'
    data['Position'] = np.select(conditions, choices)
    data.loc[
        (data['Position_number'] == 0), 'class_kind'] = 'Sample'
    data.loc[
        (data['Position_number'] == 0), 'Position'] = 'Sample'
    data.loc[
        (data['class'] == 'Equiaxed_Area'), 'class_kind'] = 'Sample'
    data.loc[
        (data['class'] == 'Equiaxed_Area'), 'Position'] = 'Sample'
    data.loc[
        (data['class'] == 'Crack') & (data['class_kind'] == 'diagonal') & (data['bbox_angle'] <= 40) | (data['class'] == 'Crack') & (data['class_kind'] == 'diagonal') & (data['bbox_angle'] >= 70), 'class_kind'] = 'hinge'
    
    
    #### 5) DataFrame으로 추가한다
    if (data.columns[-1:].to_list()[0] == 'class_kind') | (data.columns[-1:].to_list()[0] == 'Position'):        
        col1 = data.columns[:2].to_list()
        col2 = data.columns[-3:].to_list()
        col3 = data.columns[2:-3].to_list()
        new_col = col1 + col2 + col3
        data = data[new_col]
    else:
        pass
    # print(data)
    return data   
    
def find_grain_parameters(contours, hierarchy, height, width):
    
    hierarchy = tuple(map(np.array, hierarchy))  # convert hierarchy list to tuple of numpy ndarrays

    if not contours:
        print('contours가 비어있습니다.')
        return 0, 0, 0, 0, [0, 0]

    ellipses = np.array([cv2.fitEllipse(cnt) for cnt in contours], dtype=object)

    if not ellipses.any():
        print('ellipses가 모두 0으로 측정')
        return 0, 0, 0, 0, [0, 0]

    if hierarchy is not None and len(hierarchy) == 3:
        hierarchy = [hierarchy[0]]

    major_axes = np.array([ellipse[1][1] for ellipse in ellipses])
    minor_axes = np.array([ellipse[1][0] for ellipse in ellipses])
    max_major_axis = np.max(major_axes)
    min_minor_axis = np.min(minor_axes)
    
    ## 등가직경 계산-------------------------
    valid_contours_indices = [contours.index(cnt) for cnt in contours if hierarchy[0][contours.index(cnt)][3] == -1]
    centroids = np.array([[cv2.moments(contours[idx])['m10'] / cv2.moments(contours[idx])['m00'], cv2.moments(contours[idx])['m01'] / cv2.moments(contours[idx])['m00']] for idx in valid_contours_indices])

    major_minor_product = np.array([[ma * MA] for ma, MA in zip(minor_axes, major_axes)])
    total_weight = np.sum(major_minor_product)
    weighted_centroids = np.multiply(centroids, major_minor_product)
    centroid = np.sum(weighted_centroids, axis=0) / total_weight
    
    ## 유효 장축길이 계산-------------------------
    min_valid_minor_axis = min(height, width)
    max_valid_major_axis = np.sqrt(height ** 2 + width ** 2) + min(height, width)
    
    valid_major_axes_indices = np.where((major_axes >= min_valid_minor_axis) & (major_axes <= max_valid_major_axis))
    valid_major_axes = major_axes[valid_major_axes_indices]
    valid_minor_axes = minor_axes[valid_major_axes_indices]

    valid_ellipses = ellipses[valid_major_axes_indices]
    
    # print(f'{min_valid_minor_axis} ~ {max_valid_major_axis}에 {valid_major_axes}가 있음')
    
    contour_area = cv2.contourArea(contours[0])
    equivalent_diameter = np.sqrt(4 * contour_area / np.pi)
    
    if not valid_ellipses.any():
        print('해당 범위 내의 타원이 없어 원래 bbox 등가직경 계산방식으로  계산합니다.')
        min_valid_minor_axis = min(height, width)
        max_valid_major_axis = max(height, width)
        
        
        largest_ellipse_index = np.argmax(major_axes)
        largest_ellipse = ellipses[largest_ellipse_index]
        
        major_axis = max(height, width)
        minor_axis = min(height, width)

        return major_axis, minor_axis, 0, 0, centroid, largest_ellipse

    largest_valid_ellipse_index = np.argmax(valid_major_axes)
    largest_valid_ellipse = valid_ellipses[largest_valid_ellipse_index]

    return valid_major_axes[0], valid_minor_axes[0], equivalent_diameter, contour_area, centroid, largest_valid_ellipse
   
    
# model과 원본 이미지 array, filtering할 기준 class confidence score를 인자로 가지는 inference 시각화용 함수 생성. 
# 이미 inference 시 mask boolean값이 들어오므로 mask_threshold 값을 필요하지 않음. 
def get_detected_img(model, img_array, color_list, Equiaxed_score_threshold=0.6, crack_score_threshold=0.3, draw_box=True, is_print=True, save=True, surface=None):
    # 인자로 들어온 image_array를 복사.
    draw_img = img_array.copy()
    bbox_color = (0, 255, 0)
    text_color = (0, 0, 0)
    object_img = img_array.copy()
    # model과 image array를 입력 인자로 inference detection 수행하고 결과를 results로 받음.
    results = inference_detector(model, img_array)
    bbox_results = results[0]
    seg_results = results[1]
    sample_box_point = []
    CLS = []    
    crack_class_result = {
        'crack_class': None,
        'bbox_angle': None,
        'mask_angle': None,
        'box': None,
        'img': None,
        'x': None,
        'y': None,
        'w': None,
        'h': None,
        'rect': None}
    
    df = pd.DataFrame()
    total_mask = np.zeros((draw_img.shape[0], draw_img.shape[1]))
    # results 리스트를 loop를 돌면서 개별 2차원 array들을 추출하고 이를 기반으로 이미지 시각화 
    # results 리스트의 위치 index가 매핑된 Class id. 여기서는 result_ind가 class id
    # 개별 2차원 array에 오브젝트별 좌표와 class confidence score 값을 가짐. 
    for result_ind, bbox_result in enumerate(bbox_results):
        # 개별 2차원 array의 row size가 0 이면 해당 Class id로 값이 없으므로 다음 loop로 진행.
        if len(bbox_result) == 0:
            # print(f'{result_ind}는 없음')
            continue
        
        mask_array_list = seg_results[result_ind]
        
        # 해당 클래스 별로 Detect된 여러개의 오브젝트 정보가 2차원 array에 담겨 있으며, 이 2차원 array를 row수만큼 iteration해서 개별 오브젝트의 좌표값 추출. 
        for i in range(len(bbox_result)):
            # print(bbox_result[i, 4])
            # print(crack_score_threshold, Equiaxed_score_threshold)
            # 좌상단, 우하단 좌표 추출.
            if (bbox_result[i, 4] > crack_score_threshold) & (result_ind == 0) | (bbox_result[i, 4] > Equiaxed_score_threshold) & (result_ind != 0):
                left = int(bbox_result[i, 0])
                top = int(bbox_result[i, 1])
                right = int(bbox_result[i, 2])
                bottom = int(bbox_result[i, 3])
                height = bottom - top
                width = right - left
                # area = height * width
                ### 1. 크랙길이
                crack_length = np.maximum(height, width)
                
                # masking 시각화 적용. class_mask_array는 image 크기 shape의  True/False값을 가지는 2차원 array
                class_mask_array = mask_array_list[i]
                
                # Bounding box의 각도를 구하기
                # print(class_mask_array, class_mask_array.shape)
                try:
                    contours, hierarchy = cv2.findContours(class_mask_array.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
                except cv2.error:
                    print("Error: cv2.findContours() 함수가 contours를 찾지 못했습니다")
                    contours = []
                if len(contours) == 0:
                    print("Error: 주어진 이미지에서 contours를 찾지 못했습니다")
                    continue
                if contours is None:
                    # print(contours[0])
                    continue
                if len(contours[0]) < 5:
                    print('컨투어 좌표 4개 이하')
                    continue
                
                ### 1. 크랙길이(장축), 등가직경, Area계산
                max_major_axis, min_minor_axis, equivalent_diameter, area, centroid, ellipses = find_grain_parameters(contours=[contours[0]], hierarchy=hierarchy, height=height, width=width)
                if max_major_axis == 0:
                    print('길이 계산실패')
                    continue
                area = class_mask_array[class_mask_array==True].sum()
                equvalant_size = np.sqrt(4 * area / np.pi)
                if area < 10: # Crack 면적이 0인 경우 bbox 면적 적용 
                    area = height * width
                    
                # if area == 0:
                #     area = class_mask_array[class_mask_array==True].sum()
                #     equvalant_size = np.sqrt(4 * area / np.pi)


                        
                crack_length = max(height, width)
                rect = cv2.minAreaRect(contours[0])
                box = cv2.boxPoints(rect).astype(int)
                
                # 원본 image array에서 mask가 True인 영역만 별도 추출.
                masked_roi = draw_img[class_mask_array]
               
                
                # Area_mask = np.maximum(class_mask_array, total_mask)
                ### 2. 샘플면적
                # area = Area_mask[Area_mask==True].sum()
                # area = class_mask_array[class_mask_array==True].sum()
                
                # if area < 10: # Crack 면적이 0인 경우 bbox 면적 적용 
                #     area = height * width
                
                ### 3. 등가직경
                # equvalant_size = np.sqrt(area / np.pi)
                
                color_index = result_ind % len(color_list)
                color = color_list[color_index]
                if (result_ind != 1) & (result_ind != 8) | (result_ind != 9) | (result_ind != 10) | (result_ind != 11):
                    draw_img[class_mask_array] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.6 * masked_roi).astype(np.uint8)
                    total_mask = np.maximum(class_mask_array, total_mask)
                    object_mask = np.where(total_mask > 0, 255, total_mask)
                    
                    cropped_mask = object_mask[top:bottom, left:right].astype(np.uint8)
                    cropped_image = object_img[top:bottom, left:right].astype(np.uint8)
                    
                    
                # if surface == True & (result_ind == 0): #블룸표면마크로인 경우 형상분류 필요함
                if (result_ind == 0): #블룸표면마크로인 경우 형상분류 필요함
                    # print('블룸표면마크로인 경우 형상분류 필요함', cropped_mask.shape, np.unique(cropped_mask))
                    # print(cropped_mask)
                    crack_class_result =  classify_crack(img = cropped_mask)
                    # 표면 입계, 횡, 종 구분
                    crack_class = crack_class_result['crack_class']
                    # masking 각도를 구하기
                    # bbox_angle = crack_class_result['bbox_angle]
                    # mask_angle = crack_class_result['mask_angle]
                    
                if draw_box:                    
                    if result_ind == 0:
                        if crack_score_threshold < 0.13: # crack_score_threshold=0.08 정도로 낮은 경우 
                            crack_score_threshold = 0.13 # crack_score_threshold=0.13 변경 
                        if bbox_result[i, 4] > crack_score_threshold: 
                            if surface == True:
                                caption = '{}:{}: {:.4f}, ({}, {}), ({}, {}), w:{},h:{}, A:{}, L:{}'.format(labels_to_names_seq[result_ind], crack_class, bbox_result[i, 4], left, top, right, bottom, width, height, area, max_major_axis)
                            caption = '{}: {:.4f}, ({}, {}), ({}, {}), w:{},h:{}, A:{}, L:{}'.format(labels_to_names_seq[result_ind], bbox_result[i, 4], left, top, right, bottom, width, height, area, max_major_axis)
                            # cv2.rectangle(draw_img, (left, top), (right, bottom), color=bbox_color, thickness=4)
                            # cv2.ellipse(draw_img, ellipses, color=bbox_color, thickness=4)
                            cv2.drawContours(draw_img, [box], 0, color=bbox_color, thickness=4)
                            # cv2.putText(draw_img, caption, (left - 500, top - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 6)
                        else:
                            pass
                    elif (result_ind == 1) | (result_ind == 8) | (result_ind == 9) | (result_ind == 10) | (result_ind == 11):
                        caption = '{}: {:.4f}, ({}, {}), ({}, {}), w:{},h:{}, A:{}'.format(labels_to_names_seq[result_ind], bbox_result[i, 4], left, top, right, bottom, width, height, area)
                        # cv2.rectangle(draw_img, (left, top), (right, bottom), color=bbox_color, thickness=6)
                        cv2.drawContours(draw_img, [box], 0, color=bbox_color, thickness=6)
                        cv2.putText(draw_img, caption, (left - 500, top - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 6)
                        # 샘플각도에 맞게 그림을 그릴수있는 box 좌표 수집
                        sample_box_point = box.copy()
                        # print(sample_box_point)
                        
                    elif result_ind == 2:
                        caption = '{}: {:.4f}, ({}, {}), ({}, {}), size:{}'.format(labels_to_names_seq[result_ind], bbox_result[i, 4], left, top, right, bottom, equvalant_size)   
                        # cv2.rectangle(draw_img, (left, top), (right, bottom), color=bbox_color, thickness=6)
                        cv2.drawContours(draw_img, [box], 0, color=bbox_color, thickness=6)
                        # cv2.putText(draw_img, caption, (left - 500, top - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 6)
                    elif result_ind == 3:
                        caption = '{}: {:.4f}, ({}, {}), ({}, {}), A:{}'.format(labels_to_names_seq[result_ind], bbox_result[i, 4], left, top, right, bottom, area)    
                        # cv2.rectangle(draw_img, (left, top), (right, bottom), color=bbox_color, thickness=6)
                        cv2.drawContours(draw_img, [box], 0, color=bbox_color, thickness=6)
                        cv2.putText(draw_img, caption, (left + 500, top - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 6)  
                    elif (result_ind == 4) |(result_ind == 5) | (result_ind == 6) | (result_ind == 7):
                        caption = '{}: {:.4f}, ({}, {}), ({}, {}), w:{},h:{}, A:{}'.format(labels_to_names_seq[result_ind], bbox_result[i, 4], left, top, right, bottom, width, height, area)
                        # cv2.rectangle(draw_img, (left, top), (right, bottom), color=bbox_color, thickness=6)
                        cv2.drawContours(draw_img, [box], 0, color=bbox_color, thickness=6)
                        cv2.putText(draw_img, caption, (left - 500, top - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 6)


                #color를 임의 지정
                # color_index = np.random.randint(0, len(COLORS) - 1)
                # color를 class별로 지정

                # print(color_index, color)
                # apply_mask()함수를 적용시 수행 시간이 상대적으로 오래 걸림. 
                #draw_img = apply_mask(draw_img, class_mask_array, color, alpha=0.4)
                # 원본 이미지의 masking 될 영역에 mask를 특정 투명 컬러로 적용
                # if (result_ind != 1) & (result_ind != 3):

                if save:
                    cv2.imwrite('tutorial_exps/bloom_segment_model/객체모음집/CROP_MASK/{}_{}.png'.format(labels_to_names_seq[result_ind], i), cropped_mask)
                    cv2.imwrite('tutorial_exps/bloom_segment_model/객체모음집/CROP_CRACK/{}_{}.png'.format(labels_to_names_seq[result_ind], i), cropped_image)
                    
                if is_print:
                    print(caption)
    
                if result_ind == 0:
                    if crack_score_threshold < 0.13: # crack_score_threshold=0.08 정도로 낮은 경우 
                        crack_score_threshold = 0.13 # crack_score_threshold=0.13 변경 
                    if bbox_result[i, 4] > crack_score_threshold: 
                        data = pd.DataFrame({
                            'class' : [labels_to_names_seq[result_ind]],
                            'surface_crack' : [crack_class_result['crack_class']],
                            'IoU_Confidence_Score' : [bbox_result[i, 4]],
                            'left' : [left],
                            'top' : [top], 
                            'right' : [right], 
                            'bottom' : [bottom], 
                            'width' : [width], 
                            'height' : [height], 
                            'area' : [area], 
                            'bbox_angle' : [crack_class_result['bbox_angle']],
                            'mask_angle' : [crack_class_result['mask_angle']],
                            'equvalant_size' : [equvalant_size], 
                            'crack_length' : [max_major_axis]})
                else:
                    data = pd.DataFrame({
                        'class' : [labels_to_names_seq[result_ind]],
                        'surface_crack' : [crack_class_result['crack_class']],
                        'IoU_Confidence_Score' : [bbox_result[i, 4]],
                        'left' : [left],
                        'top' : [top], 
                        'right' : [right], 
                        'bottom' : [bottom], 
                        'width' : [width], 
                        'height' : [height], 
                        'area' : [area], 
                        'bbox_angle' : [crack_class_result['bbox_angle']],
                        'mask_angle' : [crack_class_result['mask_angle']],
                        'equvalant_size' : [equvalant_size], 
                        'crack_length' : [max_major_axis]})
                
                # 샘플이 무슨샘플인지 알려주는 CLASS 수집
                if (result_ind == 1) | (result_ind == 8) | (result_ind == 9) | (result_ind == 10) | (result_ind == 11):
                    CLS.append(result_ind)
    
                df = pd.concat([df, data], axis=0, ignore_index=True)
            

    return draw_img, total_mask, df, CLS, sample_box_point

def double_detected_img(model_ckpt_1, img_arr, color_list, Equiaxed_score_threshold=0.6, crack_score_threshold=0.3, draw_box=True, is_print=True, save=True):
    draw_img = img_arr.copy()
    # 결함 분류 기준 설정
    # p1 = p_list[0]
    # p2 = p_list[1]
    # p3 = p_list[2]
    # p4 = p_list[3]
    # p5 = p_list[4]
    # p6 = p_list[5]
    # p7 = p_list[6]
    # print(p_list)

    # 검출 판정 2번 하기 : model_ckpt_1(샘플), model_ckpt_2(크랙, 수축공, 등축정)
    print('검출 판정 1번째')
    first_img, total_mask, first_df, first_CLS, first_sample_box_point = get_detected_img(model_ckpt_1, draw_img, color_list=color_list, Equiaxed_score_threshold=Equiaxed_score_threshold, crack_score_threshold=Equiaxed_score_threshold, draw_box=draw_box, is_print=is_print, save=save, surface=None)
    MACRO = labels_to_names_seq[first_CLS[0]]
    # print(first_CLS)
    # print(first_CLS[0])

    print(f'{MACRO}로 샘플분류완료. 샘플분류에 맞게 2차판정진행')
    if (MACRO == 'Bloom_TB_Surface') | (MACRO == 'Bloom_Side_Surface'): # 블룸표면
        # 샘플에 맞게 모델변경
        checkpoint_file2 = 'tutorial_exps/bloom_segment_model/블룸표면/epoch_36.pth'
        model_ckpt_2 = init_detector(cfg2, checkpoint_file2, device='cuda:0')
        
        surface = True
        second_img, total_mask, second_df, second_CLS, second_sample_box_point = get_detected_img(model_ckpt_2, draw_img, color_list=color_list, Equiaxed_score_threshold=Equiaxed_score_threshold, crack_score_threshold=crack_score_threshold, draw_box=draw_box, is_print=is_print, save=save, surface=surface)
    
        # 1, 2 판정 데이터 취합하기
        df = pd.concat([second_df, first_df[(first_df['class'] == f'{MACRO}')]], axis=0, ignore_index=True)

        ### 6. 표면 : 좌우 코너으로부터 거리 : Distance_from_side
        x = np.minimum(abs(df['left'] - df[(df['class']==f'{MACRO}')]['left'].tolist()[0]), abs(df['right'] - df[(df['class']==f'{MACRO}')]['right'].tolist()[0])) 
        df['Distance_from_side'] = x + df['width']
        ### 9. 결함분류 및 위치번호 계산
        if (MACRO == 'Bloom_TB_Surface'):
            p_list =  [5.883, 20.8, 33.3, 50, 66.7, 79.2, 94.117] # Bloom_TB_Surface 기준 : 정코너 감안하여 30mm까지 정코너로 판정
            df = bloom_surface_crack_classification(df, MACRO=MACRO, p1=p_list[0], p2=p_list[1], p3=p_list[2], p4=p_list[3], p5=p_list[4], p6=p_list[5], p7=p_list[6])
        elif (MACRO == 'Bloom_Side_Surface'):  
            # 결함 분류 기준 설정
            p_list =  [2.82, 20.8, 33.3, 50, 66.7, 79.2, 97.18] # Bloom_Side_Surface 기준 : 정코너 절단 감안하여 11mm까지 정코너로 판정 
            df = bloom_surface_crack_classification(df, MACRO=MACRO, p1=p_list[0], p2=p_list[1], p3=p_list[2], p4=p_list[3], p5=p_list[4], p6=p_list[5], p7=p_list[6])
        # print(df)
        ### 크랙분류 데이터 추가
        df['class_kind']= df['surface_crack']
        # 필요없는 품질지표는 모두 None 처리
        df['Area_ratio%'] = None
        df['Equiaxed_Area_ratio%'] = None
        # df['Distance_from_surface'] = None
        df['L_shrinkage_size'] = None
        df['top_dendrite_length'] = None
        # df['centerX_ratio'] = None
        # df['centerY_ratio'] = None

    elif (MACRO == 'billet')| (MACRO == 'Bloom'): # 블룸 빌렛 단면
        if (MACRO == 'Bloom'):            
            # 샘플에 맞게 모델변경
            checkpoint_file2 = 'tutorial_exps/bloom_segment_model/블룸단면/epoch_36.pth'
            model_ckpt_2 = init_detector(cfg2, checkpoint_file2, device='cuda:0')
        if (MACRO == 'billet'):            
            # 샘플에 맞게 모델변경
            checkpoint_file2 = 'tutorial_exps/bloom_segment_model/빌렛단면/epoch_36.pth'
            model_ckpt_2 = init_detector(cfg2, checkpoint_file2, device='cuda:0')
        
        second_img, total_mask, second_df, second_CLS, second_sample_box_point = get_detected_img(model_ckpt_2, draw_img, color_list=color_list, Equiaxed_score_threshold=Equiaxed_score_threshold, crack_score_threshold=crack_score_threshold, draw_box=draw_box, is_print=is_print, save=save, surface=None)
    
        # 1, 2 판정 데이터 취합하기
        df = pd.concat([second_df, first_df[(first_df['class'] == f'{MACRO}') | (first_df['class'] == 'Equiaxed_Area') | (first_df['class'] == 'NotQuenching_Profile')]], axis=0, ignore_index=True)
        
        ### 4. 등축정률 계산 : Equiaxed_Area_ratio%
        # print(df)
        billet_area = df[(df['class'] == f'{MACRO}')]['area'].to_list()[0]
        df['Area_ratio%'] = df['area'] / billet_area * 100
        df['Equiaxed_Area_ratio%'] = df[(df['class']=='Equiaxed_Area')]['Area_ratio%'].quantile(q=0.5, interpolation='linear')
        ### 5. 표면으로부터 거리 : Distance_from_surface
        #### : (top - billet_top, bottom - billet_bottom) y 최소값과 (left - billet_left, right - billet_right) x 최소값 둘 중 가장 최소값
        x = np.minimum(abs(df['left'] - df[(df['class']==f'{MACRO}')]['left'].tolist()[0]), abs(df['right'] - df[(df['class']==f'{MACRO}')]['right'].tolist()[0])) 
        y = np.minimum(abs(df['top'] - df[(df['class']==f'{MACRO}')]['top'].tolist()[0]), abs(df['bottom'] - df[(df['class']==f'{MACRO}')]['bottom'].tolist()[0])) 
        df['Distance_from_surface'] = np.minimum(x, y)
        ### 6. 표면 : 좌우 코너으로부터 거리 : Distance_from_side
        df['Distance_from_side'] = x + df['width']
        ### 7. L방향 : 수축공 크기 : L_shrinkage_size
        df['L_shrinkage_size'] = df[df['class']=='Shrinkage']['width'].quantile(q=0.7, interpolation='linear')
        ### 8. 상부 덴드라이트 길이
        df['top_dendrite_length'] = (df[(df['class'] == 'Equiaxed_Area')]['top'].max() - df['top']).apply(lambda x: x if x > 0 else 0)
        df['top_dendrite_length'] = df[(df['class']==f'{MACRO}')]['top_dendrite_length'].tolist()[0]
        ### 9. 결함분류 및 위치번호 계산
        if (MACRO == 'Bloom'):  
            # 결함 분류 기준 설정
            p_width =  [5.883, 20.8, 33.3, 50, 66.7, 79.2, 94.117] # Bloom 너비 기준 
            p_height =  [5.883, 20.8, 40, 50, 60, 79.2, 94.117] # Bloom 높이 기준         
            df = bloom_cross_section_crack_classification(df, MACRO, p_width=p_width, p_height=p_height)
        elif (MACRO == 'billet'):  
            # 결함 분류 기준 설정
            p_list =  [8.3, 20.8, 33.3, 50, 66.7, 79.2, 91.7] # Billet 기준
            df = billet_cross_section_crack_classification(df, MACRO=MACRO, p1=p_list[0], p2=p_list[1], p3=p_list[2], p4=p_list[3], p5=p_list[4], p6=p_list[5], p7=p_list[6])
        # 필요없는 품질지표는 모두 None 처리
        df['surface_crack'] = None

    
    elif (MACRO == 'L_direction'): # L방향
        # 샘플에 맞게 모델변경
        checkpoint_file2 = 'tutorial_exps/bloom_segment_model/블룸L방향/epoch_36.pth'
        model_ckpt_2 = init_detector(cfg2, checkpoint_file2, device='cuda:0')
        
        second_img, total_mask, second_df, second_CLS, second_sample_box_point = get_detected_img(model_ckpt_2, draw_img, color_list=color_list, Equiaxed_score_threshold=Equiaxed_score_threshold, crack_score_threshold=crack_score_threshold, draw_box=draw_box, is_print=is_print, save=save, surface=None)
        # 1, 2 판정 데이터 취합하기
        df = pd.concat([second_df, first_df[(first_df['class'] == f'{MACRO}') | (first_df['class'] == 'Equiaxed_Area')]], axis=0, ignore_index=True)
        ### 6. 표면 : 좌우 코너으로부터 거리 : Distance_from_side
        x = np.minimum(abs(df['left'] - df[(df['class']==f'{MACRO}')]['left'].tolist()[0]), abs(df['right'] - df[(df['class']==f'{MACRO}')]['right'].tolist()[0])) 
        df['Distance_from_side'] = x
        ### 7. L방향 : 수축공 크기 : L_shrinkage_size
        df['L_shrinkage_size'] = df[df['class']=='Shrinkage']['width'].quantile(q=0.7, interpolation='linear')
        ### 9. 결함분류 및 위치번호 계산
        p_list =  [2.82, 20.8, 33.3, 50, 66.7, 79.2, 97.18]
        df = bloom_surface_crack_classification(df, MACRO=MACRO, p1=p_list[0], p2=p_list[1], p3=p_list[2], p4=p_list[3], p5=p_list[4], p6=p_list[5], p7=p_list[6])
        df.loc[df['class']=='Crack', 'class_kind'] = 'midway2'
        df.loc[df['class']=='Shrinkage', 'class_kind'] = 'center_SR'
        
        # 필요없는 품질지표는 모두 None 처리
        df['Area_ratio%'] = None
        df['Equiaxed_Area_ratio%'] = None
        df['Distance_from_surface'] = None
        df['surface_crack'] = None
        df['top_dendrite_length'] = None
        # df['centerX_ratio'] = None
        # df['centerY_ratio'] = None

    else: # 샘플분류판정 실패하면 종료
        pass
    
    # 데이터프레임 열 순서 고정
    df = df.reindex(columns=['class', 'surface_crack', 'Position_number', 'Position', 'class_kind',
       'IoU_Confidence_Score', 'left', 'top', 'right', 'bottom', 'width',
       'height', 'area', 'bbox_angle', 'mask_angle', 'equvalant_size',
       'crack_length', 'Area_ratio%', 'Equiaxed_Area_ratio%',
       'Distance_from_surface', 'Distance_from_side', 'L_shrinkage_size',
       'top_dendrite_length', 'left_ratio', 'right_ratio', 'top_ratio',
       'bottom_ratio', 'centerX_ratio', 'centerY_ratio'])
    
    if is_print:
        bbox_color = (0, 255, 0)
        text_color = (0, 0, 0)
        if 'NotQuenching_Profile' in first_df['class'].values:
            NotQuenching_IoU_Confidence_Score = first_df[(first_df['class'] == f'{MACRO}')].IoU_Confidence_Score.tolist()[0]
            NotQuenching_left = first_df[(first_df['class'] == 'NotQuenching_Profile')].left.tolist()[0]
            NotQuenching_top = first_df[(first_df['class'] == 'NotQuenching_Profile')].top.tolist()[0]
            NotQuenching_right = first_df[(first_df['class'] == 'NotQuenching_Profile')].right.tolist()[0]
            NotQuenching_bottom = first_df[(first_df['class'] == 'NotQuenching_Profile')].bottom.tolist()[0]
            NotQuenching_width = first_df[(first_df['class'] == 'NotQuenching_Profile')].width.tolist()[0]
            NotQuenching_height = first_df[(first_df['class'] == 'NotQuenching_Profile')].height.tolist()[0]
            NotQuenching_area = first_df[(first_df['class'] == f'NotQuenching_Profile')].area.tolist()[0]
            
            caption = '{}: {:.4f}, ({}, {}), ({}, {}), w:{},h:{}, A:{}'.format('NotQuenching_Profile', NotQuenching_IoU_Confidence_Score, NotQuenching_left, NotQuenching_top, NotQuenching_right, NotQuenching_bottom, NotQuenching_width, NotQuenching_height, NotQuenching_area)
            cv2.rectangle(second_img, (NotQuenching_left, NotQuenching_top), (NotQuenching_right, NotQuenching_bottom), color=bbox_color, thickness=6)
            cv2.putText(second_img, caption, (NotQuenching_left + 1000, NotQuenching_top - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 6)
        
        # 샘플판정모델에서 빌렛 좌표 추출
        billet_IoU_Confidence_Score = first_df[(first_df['class'] == f'{MACRO}')].IoU_Confidence_Score.tolist()[0]
        billet_left = first_df[(first_df['class'] == f'{MACRO}')].left.tolist()[0]
        billet_top = first_df[(first_df['class'] == f'{MACRO}')].top.tolist()[0]
        billet_right = first_df[(first_df['class'] == f'{MACRO}')].right.tolist()[0]
        billet_bottom = first_df[(first_df['class'] == f'{MACRO}')].bottom.tolist()[0]
        billet_width = first_df[(first_df['class'] == f'{MACRO}')].width.tolist()[0]
        billet_height = first_df[(first_df['class'] == f'{MACRO}')].height.tolist()[0]
        billet_area = first_df[(first_df['class'] == f'{MACRO}')].area.tolist()[0]

        caption = '{}: {:.4f}, ({}, {}), ({}, {}), w:{},h:{}, A:{}'.format(MACRO, billet_IoU_Confidence_Score, billet_left, billet_top, billet_right, billet_bottom, billet_width, billet_height, billet_area)
        
        cv2.drawContours(second_img, [first_sample_box_point], 0, color=bbox_color, thickness=6)
        # cv2.rectangle(second_img, (billet_left, billet_top), (billet_right, billet_bottom), color=bbox_color, thickness=6)
        cv2.putText(second_img, caption, (billet_left + 1000, billet_top - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 6)
       
    return second_img, total_mask, df

def classify_crack(img):
    crack_class = 'unknown'
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    
    # 외곽선 중 가장 큰 외곽선 선택
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect).astype(int)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    if rect[1][0] < rect[1][1]:
        mask_angle = abs(rect[2] - 90)
    else:
        mask_angle = abs(rect[2])
    
    # Bounding box 및 mask angle 계산
    bbox_angle = calc_degree(w, h)
    # mask_h, mask_w = rect[1]
    # mask_angle = calc_degree(mask_w, mask_h)
    
    if abs(bbox_angle) > 60 and abs(bbox_angle) <= 90:
        if abs(mask_angle) > 20 and abs(mask_angle) <= 60:
            crack_class = 'intergranular'
        else:
            crack_class = 'longitudnal'
    elif abs(bbox_angle) <= 20 and abs(bbox_angle) <= 20:
        if abs(mask_angle) > 20 and abs(mask_angle) <= 60:
            crack_class = 'intergranular'
        else:
            crack_class = 'transverse'
    elif abs(bbox_angle) > 20 and abs(bbox_angle) <= 60:
        if abs(mask_angle) > 80 and abs(mask_angle) <= 90:
            crack_class = 'longitudnal'
        else:
            crack_class = 'intergranular'
    
    result = {
        'crack_class': crack_class,
        'bbox_angle': bbox_angle,
        'mask_angle': mask_angle,
        'box': box,
        'img': img,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'rect': rect
    }
    
    return result

def calc_degree(width, height):
    """Bounding box 및 mask angle을 계산하여 degree로 반환합니다."""
    angle = math.atan2(height, width) * 180 / np.pi
    return angle

if __name__ == '__main__':


    # config 파일을 설정하고, 다운로드 받은 pretrained 모델을 checkpoint로 설정.
    # config_file = '/mnt/d/besteel/mmdetection/configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco.py'
    # checkpoint_file = '/mnt/d/besteel/mmdetection/checkpoints/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth'
    config_file = 'mmdetection/configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco.py'
    checkpoint_file = 'tutorial_exps/bloom_segment_model/샘플분류/epoch_36.pth'
    cfg1 = Config.fromfile(config_file)
    cfg2 = Config.fromfile(config_file)
    ################################################### Config 1 : 샘플 검출 모델조정#############################################################
    # 이미지해상도 조정 : 안함
    # defalut img_scale = (1333, 800) 
    # img_scale = (333, 200)
    # img_scale = (500, 300)
    # img_scale = (1333, 800) 

    # img_scale = (2576, 1932)

    # cfg.train_pipeline[2]['img_scale'] = img_scale
    # cfg.test_pipeline[1]['img_scale'] = img_scale
    # cfg.data.train.pipeline[2]['img_scale'] = img_scale
    # cfg.data.train.pipeline[2]['img_scale'] = img_scale
    # cfg.data.val.pipeline[1]['img_scale'] = img_scale
    # cfg.data.test.pipeline[1]['img_scale'] = img_scale

    # batch size = 2로 변경
    cfg1.auto_scale_lr = dict(enable=False, base_batch_size=16)

    #----------------------------------------------------------------
    print(
    cfg1.train_pipeline[2]['img_scale'],
    # cfg.train_pipeline[3]['img_scale'],
    cfg1.test_pipeline[1]['img_scale'],
    cfg1.data.train.pipeline[2]['img_scale'],
    # cfg.data.train.pipeline[3]['img_scale'],
    # cfg.data.train.pipeline[2]['img_scale'],
    cfg1.data.val.pipeline[1]['img_scale'],
    cfg1.data.test.pipeline[1]['img_scale'],
    cfg1.auto_scale_lr)
    #----------------------------------------------------------------

    # dataset에 대한 환경 파라미터 수정. 
    cfg1.dataset_type = 'BloomDataset'
    # cfg.data.train.dataset.type  = 'BilletDataset'
    # cfg.dataset.type = 'BilletDataset'
    cfg1.data_root = 'coco_bloom/'

    # del cfg['data']['train']['times']

    # train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
    cfg1.data.train.type = 'BloomDataset'
    cfg1.data.train.data_root = 'coco_bloom/'
    cfg1.data.train.ann_file = 'train_coco.json'
    cfg1.data.train.img_prefix = 'train'

    # cfg.data.train.dataset.ann_file = 'train_coco.json'
    # cfg.data.train.dataset.img_prefix = 'train'

    # cfg.data.train.seg_prefix = ''

    cfg1.data.val.type = 'BloomDataset'
    cfg1.data.val.data_root = 'coco_bloom/'
    cfg1.data.val.ann_file = 'val_coco.json'
    cfg1.data.val.img_prefix = 'val'
    # cfg.data.val.seg_prefix = ''
    # cfg.data.val.ins_ann_file = None

    cfg1.data.test.type = 'BloomDataset'
    cfg1.data.test.data_root = 'coco_bloom/'
    cfg1.data.test.ann_file = 'val_coco.json'
    cfg1.data.test.img_prefix = 'val'
    # cfg.data.test.seg_prefix = ''
    # cfg.data.test.ins_ann_file = None

    # class의 갯수 수정. 
    cfg1.model.roi_head.bbox_head.num_classes = 12
    cfg1.model.roi_head.mask_head.num_classes = 12
    # mask2transformer의 num_classes
    # cfg.num_things_classes  =  4
    # cfg.num_stuff_classes  =  0
    # cfg.num_classes  =  cfg.num_things_classes  +  cfg.num_stuff_classes

    # pretrained 모델
    cfg1.load_from = checkpoint_file

    # 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정. 

    cfg1.work_dir = 'tutorial_exps/bloom_segment_model'

    # 학습율 변경 환경 파라미터 설정. 
    cfg1.optimizer.lr = 0.02 / 8
    # cfg.lr_config.warmup = None
    cfg1.log_config.interval = 10

    # CocoDataset의 경우 metric을 bbox로 설정해야 함.(mAP아님. bbox로 설정하면 mAP를 iou threshold를 0.5 ~ 0.95까지 변경하면서 측정)
    cfg1.evaluation.metric = ['bbox', 'segm']
    cfg1.evaluation.interval = 2
    cfg1.checkpoint_config.interval = 2

    # epochs 횟수는 36으로 증가 
    cfg1.runner.max_epochs = 36 
    # cfg.interval = 10
    # cfg.runner.max_iters = 36
    # cfg.max_iters = 36
    # 두번 config를 로드하면 lr_config의 policy가 사라지는 오류로 인하여 설정. 
    cfg1.lr_config.policy = 'step'
    # Set seed thus the results are more reproducible
    cfg1.seed = 0
    set_random_seed(0, deterministic=False)
    cfg1.gpu_ids = range(1)

    # ConfigDict' object has no attribute 'device 오류 발생시 반드시 설정 필요. https://github.com/open-mmlab/mmdetection/issues/7901
    cfg1.device='cuda'

    ################################################### Config 2 : 크랙, 수축공, 등축정 검출 모델 조정#############################################################
    # 이미지해상도 조정 : 안함
    # defalut img_scale = (1333, 800) 
    # img_scale = (333, 200)
    # img_scale = (500, 300)
    # img_scale = (1333, 800) 

    img_scale = (2576, 1932)

    cfg2.train_pipeline[2]['img_scale'] = img_scale
    cfg2.test_pipeline[1]['img_scale'] = img_scale
    cfg2.data.train.pipeline[2]['img_scale'] = img_scale
    cfg2.data.train.pipeline[2]['img_scale'] = img_scale
    cfg2.data.val.pipeline[1]['img_scale'] = img_scale
    cfg2.data.test.pipeline[1]['img_scale'] = img_scale

    # batch size = 2로 변경
    cfg2.auto_scale_lr = dict(enable=False, base_batch_size=16)

    #----------------------------------------------------------------
    print(
    cfg2.train_pipeline[2]['img_scale'],
    # cfg.train_pipeline[3]['img_scale'],
    cfg2.test_pipeline[1]['img_scale'],
    cfg2.data.train.pipeline[2]['img_scale'],
    # cfg2.data.train.pipeline[3]['img_scale'],
    # cfg2.data.train.pipeline[2]['img_scale'],
    cfg2.data.val.pipeline[1]['img_scale'],
    cfg2.data.test.pipeline[1]['img_scale'],
    cfg2.auto_scale_lr)
    #----------------------------------------------------------------

    # dataset에 대한 환경 파라미터 수정. 
    cfg2.dataset_type = 'BloomDataset'
    # cfg.data.train.dataset.type  = 'BilletDataset'
    # cfg.dataset.type = 'BilletDataset'
    cfg2.data_root = 'coco_bloom/'

    # del cfg['data']['train']['times']

    # train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
    cfg2.data.train.type = 'BloomDataset'
    cfg2.data.train.data_root = 'coco_bloom/'
    cfg2.data.train.ann_file = 'train_coco.json'
    cfg2.data.train.img_prefix = 'train'

    # cfg.data.train.dataset.ann_file = 'train_coco.json'
    # cfg.data.train.dataset.img_prefix = 'train'

    # cfg.data.train.seg_prefix = ''

    cfg2.data.val.type = 'BloomDataset'
    cfg2.data.val.data_root = 'coco_bloom/'
    cfg2.data.val.ann_file = 'val_coco.json'
    cfg2.data.val.img_prefix = 'val'
    # cfg.data.val.seg_prefix = ''
    # cfg.data.val.ins_ann_file = None

    cfg2.data.test.type = 'BloomDataset'
    cfg2.data.test.data_root = 'coco_bloom/'
    cfg2.data.test.ann_file = 'val_coco.json'
    cfg2.data.test.img_prefix = 'val'
    # cfg.data.test.seg_prefix = ''
    # cfg.data.test.ins_ann_file = None

    # class의 갯수 수정. 
    cfg2.model.roi_head.bbox_head.num_classes = 12
    cfg2.model.roi_head.mask_head.num_classes = 12
    # mask2transformer의 num_classes
    # cfg.num_things_classes  =  4
    # cfg.num_stuff_classes  =  0
    # cfg.num_classes  =  cfg.num_things_classes  +  cfg.num_stuff_classes

    # pretrained 모델
    cfg2.load_from = checkpoint_file

    # 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정. 

    cfg2.work_dir = 'tutorial_exps/bloom_segment_model'

    # 학습율 변경 환경 파라미터 설정. 
    cfg2.optimizer.lr = 0.02 / 8
    # cfg.lr_config.warmup = None
    cfg2.log_config.interval = 10

    # CocoDataset의 경우 metric을 bbox로 설정해야 함.(mAP아님. bbox로 설정하면 mAP를 iou threshold를 0.5 ~ 0.95까지 변경하면서 측정)
    cfg2.evaluation.metric = ['bbox', 'segm']
    cfg2.evaluation.interval = 2
    cfg2.checkpoint_config.interval = 2

    # epochs 횟수는 36으로 증가 
    cfg2.runner.max_epochs = 36 
    # cfg.interval = 10
    # cfg.runner.max_iters = 36
    # cfg.max_iters = 36
    # 두번 config를 로드하면 lr_config의 policy가 사라지는 오류로 인하여 설정. 
    cfg2.lr_config.policy = 'step'
    # Set seed thus the results are more reproducible
    cfg2.seed = 0
    set_random_seed(0, deterministic=False)
    cfg2.gpu_ids = range(1)

    # ConfigDict' object has no attribute 'device 오류 발생시 반드시 설정 필요. https://github.com/open-mmlab/mmdetection/issues/7901
    cfg2.device='cuda'
    
    print(
    cfg1.train_pipeline[2]['img_scale'],
    # cfg.train_pipeline[3]['img_scale'],
    cfg1.test_pipeline[1]['img_scale'],
    cfg1.data.train.pipeline[2]['img_scale'],
    # cfg.data.train.pipeline[3]['img_scale'],
    # cfg.data.train.pipeline[2]['img_scale'],
    cfg1.data.val.pipeline[1]['img_scale'],
    cfg1.data.test.pipeline[1]['img_scale'],
    cfg1.auto_scale_lr)

    print(
    cfg2.train_pipeline[2]['img_scale'],
    # cfg.train_pipeline[3]['img_scale'],
    cfg2.test_pipeline[1]['img_scale'],
    cfg2.data.train.pipeline[2]['img_scale'],
    # cfg.data.train.pipeline[3]['img_scale'],
    # cfg.data.train.pipeline[2]['img_scale'],
    cfg2.data.val.pipeline[1]['img_scale'],
    cfg2.data.test.pipeline[1]['img_scale'],
    cfg2.auto_scale_lr)
    
    ################################################## mAP계산 #############################################################
    # file_path = 'tutorial_exps/billet_segment_model/mask_rcnn_x101_64x4d_fpn_1x_coco/None.log.json'
    # train, val = evaluate_mAP(file_path)
    # visualization(val, ymin=0, ymax=0.45)
    ################################################### inference #############################################################
    # checkpoint_file = '/mnt/d/besteel/tutorial_exps/billet_segment_model/mask_rcnn_x101_64x4d_fpn_1x_coco/epoch_36.pth'
    # checkpoint_file = 'tutorial_exps/billet_segment_model/mask_rcnn_x101_64x4d_fpn_1x_coco/epoch_36.pth'
    
    
    # checkpoint 저장된 model 파일을 이용하여 모델을 생성, 이때 Config는 위에서 update된 config 사용. 
    checkpoint_file = 'tutorial_exps/bloom_segment_model/샘플분류/epoch_36.pth'
    model_ckpt_1 = init_detector(cfg1, checkpoint_file, device='cuda:0')
    # checkpoint_file2 = 'tutorial_exps/bloom_segment_model/블룸표면/epoch_36.pth'
    # checkpoint_file2 = 'tutorial_exps/bloom_segment_model/epoch_10.pth'
    # model_ckpt_2 = init_detector(cfg2, checkpoint_file2, device='cuda:0')

    import matplotlib
    # %matplotlib qt # 창모드
    # %matplotlib inline # 주피터 노트북 모드

    # image_path = 'bloom/블룸표면/125.jpg'
    # image_path = 'bloom/블룸표면/258.jpg'
    # image_path = 'bloom/블룸표면/292.jpg'
    # image_path = 'bloom/블룸표면/130.jpg'
    # image_path = 'bloom/블룸L방향/163.jpg'
    # image_path = 'bloom/블룸L방향/200.jpg'
    # image_path = 'bloom/블룸단면/211.jpg'
    # image_path = 'bloom/빌렛단면/111.JPG'
    # image_path = 'bloom/블룸단면/288.jpg'
    # image_path = sys.argv[1]
    image_path = 'result/빌렛BC급/SPS4_B76474_406_상면.jpg'
    
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
    COLORS = list(
        [[0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [80, 70, 180],
        [250, 80, 190],
        [245, 145, 50],
        [70, 150, 250]] )
    
    # 결함 분류 기준
    # p_list = [15, 33, 50, 66, 83] # Bloom 기준
    # p_list = [20, 30, 50, 60, 80] # 
    # p_list = [25, 30, 50, 70, 75] # (예상) Billet 기준
    # print(p_list)
    # torch.cuda.empty_cache()
    
    # img_arr = cv2.cvtColor(cv2.imread('{}'.format(image_path)), cv2.COLOR_BGR2RGB)
    img_arr = cv2.imread('{}'.format(image_path))
    detected_img, total_mask, data = double_detected_img(model_ckpt_1=model_ckpt_1,
                                                        img_arr=img_arr, 
                                                        color_list=COLORS,
                                                        Equiaxed_score_threshold=0.6,
                                                        crack_score_threshold=0.15, # crack_score_threshold<0.13이면 crack_score_threshold=0.13으로 고정함
                                                        draw_box=True,
                                                        is_print=True,
                                                        save=True)
    # detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(20,20))
    # plt.imshow(detected_img)
    # plt.title('segmented image')
    # print(data)
    total_mask = np.where(total_mask>0, 255, total_mask)
    
    # detected_img.save("result/detected_image.png")
    cv2.imwrite("result/detected_image.png", detected_img)
    cv2.imwrite("result/detected_mask.png", total_mask)
    data.to_excel('result/{}.xlsx'.format('macro_index'), index=False)
    print('total inference 시간: {}초'.format(np.round(time.time() - start_time), 1))
    # plt.show()
    # cv2.namedWindow('detected_image', cv2.WINDOW_NORMAL)
    # cv2.imshow('detected_image', detected_img) 
    cv2.waitKey()
    cv2.destroyAllWindows()


