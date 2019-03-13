# ChangeTextInImage
Case Converter
reference
1. https://github.com/Josh-Payne/cs230/blob/master/Alphanumeric-Augmented-CNN/augmented-cnn.py
2. https://www.kaggle.com/crawford/emnist/version/3

## What you should do.
```
pip install keras
pip install opencv-python
pip install --upgrade h5py
pip install imutils
pip install tensorflow
pip install --upgrade numpy
```

## cautions
We work this on Anaconda3
python 3.6.8
opencv 4.0.0
keras 2.2.4


## folder information
1. ChangeTextInImage/image/
  
  (1) oriImage - 원본 이미지
  
  (2) trainImage - 자르기, 그레이스케일, 이진화 이미지
  
  (3) trainImage2 - 자르기, 그레이스케일, 이진화, laplacian, resize 이미지

2. ChangeTextInImage/.ipynb_checkpoints/
  
  (1) 주피터 파일 체크포인트 저장

3. ChangeTextInImage/text_recog_module/
  
  (1) negative - negative Image (CNN) label = 1
  
  (2) positive - positive Image (CNN) label = 0
  
4. ChangeTextInImage/target_img_extract_module/
  
  (1) 이미지 잘라내는 파이썬 파일 저장되있음


## file information
@@ .ipynb - 주피터 파일

@@ None .md file information


1. ChangeTextInImage/
  
  (1) module_link.py - 원본 이미지를 넣으면 말풍선을 찾아서 keras 로 학습시킨 모델을 통과시켜 판별해 반환하는 파이썬 파일
  
  (2) reference.txt - 코드를 만드는데 도움 받은 페이지 주소 목록
  
2. ChangeTextInImage/text_recog_module/
  
  (1) jpg.py - keras 사용 모델 훈련 파이썬 파일
  
3. ChangeTextInImage/target_img_extract_module/
  (1) SaveImage.py - 원본 이미지를 훈련 모델에 넣기 위해 가공 후 새로운 이미지로 trainImage, trainImage2에 저장하는 파이썬 파일
