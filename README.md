# macro-instance-segmentation
instance segmentation 중 Mask RCNN을 활용한 cast bloom의 macro crack detection


### MMDetection 설치
* pip install mmcv-full로 mmcv를 설치(약 10분 정도의 시간이 소요)
* 실습코드는 pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html 로 변경(설치에 12초 정도 걸림. 2022.07.27).
* cuda, touch 버전은 호환 확인 할 것(cu113, torch1.12.0 부분 변경)

```python jupyter notebook
!nvcc -V
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2019 NVIDIA Corporation
# Built on Sun_Jul_28_19:07:16_PDT_2019
# Cuda compilation tools, release 10.1, V10.1.243

!pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
!pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html

!git clone https://github.com/open-mmlab/mmdetection.git
!git clone https://github.com/open-mmlab/mmcv.git

!pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
!git clone https://github.com/open-mmlab/mmdetection.git
!cd mmdetection; python setup.py install
```

# 확인하기
```python
import torch
print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from mmdet.apis import init_detector, inference_detector
import mmdet
import mmcv
print(mmcv.__version__)
print(mmdet.__version__)
```

# 이후 git clone으로 다운로드
