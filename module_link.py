
# coding: utf-8

# In[41]:


import math
import cv2
import numpy as np
import imutils
import os
from keras.models import model_from_json
from PIL import Image     #pip install pillow
from pytesseract import * #pip install pytesseract
import configparser


# In[87]:


#사각형
class rectang:
    x1=0
    y1=0
    x2=0
    y2=0
    live = 1
    def __init__(self, x1, y1, x2, y2):
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2

#허프 변환
def removeVerticalLines(img, limit):
    lines = None
    threshold = 100
    minLength = 60
    lineGap = 10
    rho = 1
    lines = cv2.HoughLinesP(img, rho, np.pi/180, threshold, minLength, lineGap)
    if(lines is not None):   # lines  이 비지 않았을때만 실행한다.
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                gapY = abs(y2-y1)
                gapX = abs(x2-x1)
                if(gapY>limit or gapX>limit and limit>0):
                    cv2.line(img, (x1,y1), (x2,y2), (0, 0, 0), 3)

def change1(img):
    temp_img = img.copy()
    temp_img = cv2.bilateralFilter(temp_img,9,75,75)
    #노이즈 제거 위한 커널(erode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    erode = cv2.erode(temp_img, kernel, iterations=1)
    #이미지 grayscale
    gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
    #global 이진화
    ret1, th = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    canny = cv2.Canny(th,180,250, apertureSize = 5)
    #직선 제거
    removeVerticalLines(canny, 70)
    return canny


# 작은 사각형 합쳐서 큰 사각형 합치는 함수
def combineRectang(rect_List):
    avr_height = 0
    #   사각형 내부 사각형 live = 0 만듬
    for r1 in range(len(rect_List)):
        for r2 in range(len(rect_List)):
            if ((rect_List[r1].x1 < rect_List[r2].x1) and (rect_List[r1].x2 >= rect_List[r2].x2)):
                if ((rect_List[r1].y1 < rect_List[r2].y1) and (rect_List[r1].y2 >= rect_List[r2].y2)):
                    rect_List[r2].live = 0
    # live = 0 인 인덱스 제거
    temp_List = []
    for x in rect_List:
        if (x.live == 1):
            temp_List.append(x);
    rect_List = temp_List

    # 버블 정렬로 사각형들 왼쪽에서 오른쪽순으로 정렬함(가장 왼쪽 사각형이 0)
    for i in range(len(rect_List)):
        for j in range(0, len(rect_List) - (i + 1)):
            if (rect_List[j].x1 > rect_List[j + 1].x1):
                temp_rect = rect_List[j]
                rect_List[j] = rect_List[j + 1]
                rect_List[j + 1] = temp_rect
    # 높이 평균 1번
    for x in rect_List:
        avr_height = avr_height + abs(x.y2 - x.y1)
    avr_height = avr_height / len(rect_List)

    #     오른쪽으로 인접한 사각형 구할때 필요한 count 값 만들기
    for i in range(len(rect_List) - 1):
        for j in range((i + 1), len(rect_List)):
            plate_width = 0
            dif_x = abs(rect_List[i].x2 - rect_List[j].x1)  # 문자 하나의 끝과 다음 사각형 문자 시작 사이의 x거리차이
            dif_y = abs(rect_List[i].y1 - rect_List[j].y1)
            # 첫번째 합쳐지는 j 찾음 = k
            if (rect_List[j].live == 1 and dif_x < avr_height * 1.5 and dif_y < avr_height * 0.5):
                rect_List[j].live = 0
                plate_width = abs(rect_List[i].x2 - rect_List[j].x2)  # i 끝과 j 끝의 x좌표 차이
            rect_List[i].x2 = rect_List[i].x2 + plate_width

    # 리스트에서 live ==0 인 인덱스 제거
    rect_List2 = []
    for x in rect_List:
        if (x.live == 1):
            rect_List2.append(x)
    # 가로 사각형 합치기 끝

    # 버블 정렬로 사각형들 위에서 아래순으로 정렬함(가장 위 사각형이 0)
    for i in range(len(rect_List2)):
        for j in range(0, len(rect_List2) - (i + 1)):
            if (rect_List2[j].y1 > rect_List2[j + 1].y1):
                temp_rect = rect_List2[j]
                rect_List2[j] = rect_List2[j + 1]
                rect_List2[j + 1] = temp_rect

    # 높이 평균 2번
    avr_height = 0
    for x in rect_List2:
        avr_height = avr_height + abs(x.y2 - x.y1)
    avr_height = avr_height / len(rect_List2)
    print('avr_height : ', avr_height)
    # 세로 사각형 합치기
    for i in range(len(rect_List2) - 1):
        # 사각형 i와 사각형 j의 가로, 세로 차이
        dif_x = 0
        dif_y = 0
        # 맨위측 과 맨아래측 사각형 x, y 좌표 차이
        plate_height = 0
        # 바로 오른쪽 사각형과의 x 좌표 차이를 구하는 변수 구해야함
        k = 0
        # k와 j의 x와 y 좌표 차이 (바로 맞닿은 사각형끼리의 차이 k와 j는 계속 변함)
        side_x = 0
        side_y = 0
        for j in range((i + 1), len(rect_List2)):
            plate_height = 0
            switch = False
            dif_x = abs(rect_List2[i].x1 - rect_List2[j].x1)  # 가로 합치는 부분과 다름, i와 j의 사각형 시작 x좌표 차이
            dif_x2 = abs(rect_List2[i].x2 - rect_List2[j].x2)
            dif_y = abs(rect_List2[i].y2 - rect_List2[j].y1)  # 가로 합치는 부분과 다름, i의 끝과 j의 시작 사이의 y좌표 차이

            if (rect_List2[i].x1 >= rect_List2[j].x1 and rect_List2[i].x2 <= rect_List2[j].x2):
                switch = True
            elif (rect_List2[i].x1 <= rect_List2[j].x1 and rect_List2[i].x2 >= rect_List2[j].x2):
                switch = True
            # 첫번째 합쳐지는 j 찾음 = k
            elif (dif_x < avr_height * 2 and dif_x2 < avr_height * 2):
                switch = True
            if (switch and dif_y < avr_height and rect_List2[j].live):
                rect_List2[j].live = 0
                plate_height = rect_List2[j].y2 - rect_List2[i].y2  # i 끝과 j 끝의 y좌표 차이
                if (rect_List2[i].x1 > rect_List2[j].x1):
                    rect_List2[i].x1 = rect_List2[j].x1
                if (rect_List2[i].x2 < rect_List2[j].x2):
                    rect_List2[i].x2 = rect_List2[j].x2
            rect_List2[i].y2 = rect_List2[i].y2 + plate_height
    # 세로 사각형 합치기 끝

    # 리스트에서 live ==0 인 인덱스 제거 제거
    rect_List3 = []
    for x in rect_List2:
        if (x.live == 1):
            rect_List3.append(x);
    for r1 in range(len(rect_List3)):
        for r2 in range(len(rect_List3)):
            if (r1 == r2):
                continue
            if ((rect_List3[r1].x1 <= rect_List3[r2].x1) and (rect_List3[r1].x2 >= rect_List3[r2].x2)):
                if ((rect_List3[r1].y2 - rect_List3[r1].y1) > (rect_List3[r2].y2 - rect_List3[r2].y1)):
                    rect_List3[r2].live = 0
            if ((rect_List3[r1].y1 <= rect_List3[r2].y1) and (rect_List3[r1].y2 >= rect_List3[r2].y2)):
                if ((rect_List3[r1].x2 - rect_List3[r1].x1) > (rect_List3[r2].x2 - rect_List3[r2].x1)):
                    rect_List3[r2].live = 0
    rect_List4 = []
    for x in rect_List3:
        if (x.live == 1):
            rect_List4.append(x);
    return rect_List4


# In[88]:


#불러올 이미지 주소 가져오기
path = '12.jpg'


# In[89]:


#텐서플로우에 전달할 이미지를 저장할 배열
image_List=[]
image_List2=[]
#rectangle 배열
rect_List = []

#이미지 변수에 저장
src = cv2.imread(path, cv2.IMREAD_UNCHANGED)

#drawContours 가 원본이미지를 변경하기에 이미지 복사
img1 = src.copy() #처음 Contours 그려짐
img2 = src.copy() #Rectangle Contours 그려짐
img3 = src.copy() #정리후 Rectangle Contours 그려짐


# In[90]:


#CannyEdge
canny = change1(src)

#Contours 찾음
contours, hierachy = cv2.findContours(canny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#그림에 Contours 그림
img1=cv2.drawContours(img1, contours, -1, (0,255,0),1)

#Contours를 사각형으로 만듬
for cnt in contours:

    #크기 작은 사격형 Contours 그리지 않음
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    if (w<70) or (h<50):
        continue
    #rectangle 좌표들 배열에 저장
    rect_List.append(rectang(x, y, x+w, y+h))

#사각형 내부의 사각형 제거. 가로,세로 좌표가 다른 사각형 내부에 포함되면 그려지지않게함
for r1 in rect_List:
    switch = True
    for r2 in rect_List:
            continue
    #해당 될시 그리는부분 스킵
    if(switch):
        img2 = cv2.rectangle(img2,(r1.x1, r1.y1),(r1.x2, r1.y2),(0,255,0),1)

        #배열에 텐서플로우에 전달할 이미지 저장
        dst = src.copy()
        dst = src[r1.y1:r1.y2, r1.x1:r1.x2]
        dst_gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
        dst_ret, dst_gray = cv2.threshold(dst_gray, 127,255,cv2.THRESH_BINARY)
        dst_laplacian = cv2.Laplacian(dst_gray, cv2.CV_8U)
        dst_laplacian = cv2.resize(dst_laplacian, dsize=(100, 100), interpolation=cv2.INTER_AREA)
        image_List.append(dst_gray)
        image_List2.append(dst_laplacian)

json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

#모델에 맞게 inputdata를 reform
for i in range(len(image_List)):
    image_List[i] = cv2.resize(image_List[i], (200, 200))
    image_List[i] = image_List[i] / 255
    
image_List = np.asarray(image_List)
image_List = image_List.reshape(len(image_List), 200, 200, 1)


# In[91]:


#모델에 image를 적용하여 predict class 출력
y = loaded_model.predict_classes(image_List)
print(y)

#모델을 거쳐 말풍선인 이미지만 저장할 배열 생성
text_rect_List = []
text_image_List = []


# In[92]:


# predict내부의 값을 가지고 말풍선이 담긴 이미지 저장
for i in range(len(y)):
    if y[i] == 0:
        dst = src.copy()
        dst = src[rect_List[i].y1:rect_List[i].y2, rect_List[i].x1:rect_List[i].x2]
        text_image_List.append(dst)
print(text_image_List)


# In[93]:


#텐서플로우에 전달할 이미지를 저장할 배열
combine_image_List = []
#rectangle 배열
combine_rect_List = []
#이미지 변수에 저장
for image in text_image_List:
    combine_src = image
    #drawContours 가 원본이미지를 변경하기에 이미지 복사
    combine_img1 = combine_src.copy() #처음 Contours 그려짐
    combine_img2 = combine_src.copy() #Rectangle Contours 그려짐
    #CannyEdge
    combine_canny = change1(combine_src)

    #Contours 찾음
    combine_contours, combine_hierachy = cv2.findContours(combine_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #그림에 Contours 그림
    combine_img1=cv2.drawContours(combine_img1, combine_contours, -1, (0,255,0),1)
    #Contours를 사각형으로 만듬
    for cnt in combine_contours:
        
        # 정해진 크기가 아닌 사격형 Contours 그리지 않음
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        if (h<10) or (h>100) or (w>100):
            continue
        if (aspect_ratio>1.5) and(aspect_ratio<=0.2) :
            continue
    
        #rectangle 좌표들 배열에 저장
        combine_rect_List.append(rectang(x, y, x+w, y+h))
    
        #######중심함수
    combine_rect_List=combineRectang(combine_rect_List)
    
    for o3 in range(len(combine_rect_List)):
        combine_img2 = cv2.rectangle(combine_img2,(combine_rect_List[o3].x1, combine_rect_List[o3].y1),(combine_rect_List[o3].x2, combine_rect_List[o3].y2),(0,255,0),1)
        
        #배열에 텐서플로우에 전달할 이미지 저장
        combine_dst = combine_src.copy()
        combine_dst = combine_src[combine_rect_List[o3].y1:combine_rect_List[o3].y2, combine_rect_List[o3].x1:combine_rect_List[o3].x2]
        
        print('세로',combine_rect_List[o3].y2-combine_rect_List[o3].y1, '가로', combine_rect_List[o3].x2-combine_rect_List[o3].x1)
        combine_image_List.append(combine_dst)

    #for i in range(len(combine_image_List)):
    #    cv2.imshow("img"+str(i),image_List[i])
    #사각형으로 변현한 Contours 출력
    cv2.imshow("img", combine_img2)

# Config Parser 초기화
config = configparser.ConfigParser()
# Config File 읽기
config.read(os.path.dirname(os.path.realpath(__file__)) + os.sep + 'envs' + os.sep + 'property.ini')


# 이미지 -> 문자열 추출
def ocrToStr(img, outTxtPath, fileName, lang='eng'):  # 디폴트는 영어로 추출
    # 이미지 경로
    fileName = str(fileName)
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, bw_img = cv2.threshold(bw_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("bw_img" + fileName, bw_img)
    bw_img = Image.fromarray(bw_img)
    print("------------")
    txtName = os.path.join(outTxtPath, fileName.split('.')[0])
    # 추출(이미지파일, 추출언어, 옵션)
    # preserve_interword_spaces : 단어 간격 옵션을 조절하면서 추출 정확도를 확인한다.
    # psm(페이지 세그먼트 모드 : 이미지 영역안에서 텍스트 추출 범위 모드)
    # psm 모드 : https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
    outText = pytesseract.image_to_string(bw_img)

    print('+++ OCT Extract Result +++')
    print('Extract FileName ->>> : ', fileName, ' : <<<-')
    print('\n\n')
    # 출력
    outText = outText.lower()
    print(outText)
    # 추출 문자 텍스트 파일 쓰기
    strToTxt(txtName, outText)

    # 문자열 -> 텍스트파일 개별 저장


def strToTxt(txtName, outText):
    with open(txtName + '.txt', 'w', encoding='utf-8') as f:
        f.write(outText)

    # 메인 시작


if __name__ == "__main__":

    # 텍스트 파일 저장 경로
    outTxtPath = os.path.dirname(os.path.realpath(__file__)) + config['Path']['OcrTxtPath']
    # OCR 추출 작업 메인
    print("사진 갯수 = ", len(combine_image_List))
    for i in range(len(combine_image_List)):
        # 한글+영어 추출(kor, eng , kor+eng)
        ocrToStr(combine_image_List[i], outTxtPath, i, 'eng')

cv2.waitKey(0)
cv2.destroyAllWindows()
