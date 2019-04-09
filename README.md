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
pip install pillow
pip install pytesseract

```

## cautions
We work this on Anaconda3
python 3.6.8
opencv 4.0.0
keras 2.2.4


## folder information
1. ChangeTextInImage/image/

>>1.1 oriImage - 원본 이미지

>>1.2 trainImage - 자르기, 그레이스케일, 이진화 이미지

>>1.3 trainImage2 - 자르기, 그레이스케일, 이진화, laplacian, resize 이미지


2. ChangeTextInImage/.ipynb_checkpoints/

>>2.1 주피터 파일 체크포인트 저장


3. ChangeTextInImage/text_recog_module/

>>3.1 negative - negative Image (CNN) label = 1

>>3.2 positive - positive Image (CNN) label = 0


4. ChangeTextInImage/target_img_extract_module/

>>4.1 이미지 잘라내는 파이썬 파일 저장되있음

5. ChangeTextInImage/text_combine/

>>5.1 말풍선을 자른 이미지에서 말풍선 내부의 문장을 문단별로 나눠서 새로 저장시켜줌




## file information
>>.ipynb - 주피터 파일

>>None .md file information


1. ChangeTextInImage/

>>1.1 module_link.py - 원본 이미지를 넣으면 말풍선을 찾고 keras로 학습시킨 모델을 통과시켜 옳은 말풍선인지 판별 후 말풍선 이미지를 반환하는 파이썬 파일
>>OpenCV 자르기 -> keras -> 사각형합치기

>>1.2 reference.txt - 코드를 만드는데 도움 받은 페이지 주소 목록

2. ChangeTextInImage/text_recog_module/

>>2.1 jpg.py - keras를 이용하여 1차적으로 말풍선을 개괄적으로 찾아내는 모델


3. ChangeTextInImage/target_img_extract_module/

>>3.1 SaveImage.py - 원본 이미지를 훈련 모델에 넣기 위해 가공 후 새로운 이미지로 trainImage, trainImage2에 저장하는 파이썬 파일

4. ChangeTextInImage/text_combine/

>>4.1 CombineWord.ipynb - 문현호가 짠 사각형 합치는 알고리즘(사용예정없음)

>>4.2 CombineWord.py - 프로토타입 사각형 알고리즘(구버전)

>>4.3 CombineWord2.py - 완성된 사각형


5. ChangeTextInImage/text_Occur_module/

>>4.1 text_Occur_module.py - keras를 이용하여 2차적으로 말풍선 내 문자열을 확인하고 filtering 하는 모델
