
# coding: utf-8

# In[328]:


import math
import cv2
import numpy as np
import imutils


# In[515]:


#사각형
#x1, x2 = 사각형의 좌우 x좌표
#y1, y2 = 사각형 위아래 y좌표
#count = 우측으로 연결된 사각형 개수
class rectang:
    x1=0
    y1=0
    x2=0
    y2=0
    count=0
    w = 0 # 너비
    h = 0 # 높이
    live = 1 # 0 이 되면 지울 사각형
    def __init__(self, x1, y1, x2, y2):
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2
        self.w = x2 - x1
        self.h = y2 - y1
#받은 이미지 확률 높이게하는 필터
def change1(img):
    temp_img = img.copy()
#     temp_img = cv2.GaussianBlur(temp_img, (5,5), 0)
    
    #노이즈 제거 위한 커널(erode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,1))
    #이미지 grayscale
    gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
    dilation = cv2.erode(gray, kernel, iterations=2)

    # 이진화
    ret1, th = cv2.threshold(dilation,127,255,cv2.THRESH_BINARY)
    canny = cv2.Canny(th,180,250, apertureSize = 5) 

    return canny
# 작은 사각형 합쳐서 큰 사각형 합치는 함수
def combineRectang(rect_List):

    #   사각형 내부 사각형 live = 0 만듬
    for r1 in range(len(rect_List)):
        for r2 in range(len(rect_List)):
            if ((rect_List[r1].x1<rect_List[r2].x1) and (rect_List[r1].x2 >= rect_List[r2].x2)) :
                if ((rect_List[r1].y1 < rect_List[r2].y1) and (rect_List[r1].y2 >= rect_List[r2].y2)) :
                    rect_List[r2].live = 0
    # live = 0 인 인덱스 제거
    temp_List = []
    for x in rect_List:
        if(x.live==1):
            temp_List.append(x);
    rect_List = temp_List
    
    #버블 정렬로 사각형들 왼쪽에서 오른쪽순으로 정렬함(가장 왼쪽 사각형이 0)
    for i in range(len(rect_List)):
        for j in range(0,len(rect_List)-(i+1)):
            if (rect_List[j].x1 > rect_List[j+1].x1):
                temp_rect = rect_List[j]
                rect_List[j] = rect_List[j+1]
                rect_List[j+1] = temp_rect
    
#     오른쪽으로 인접한 사각형 구할때 필요한 count 값 만들기
    for i in range(len(rect_List)-1):
        # 사각형 i와 사각형 j의 가로, 세로 차이
        dif_x = 0
        dif_y = 0
        #맨좌측과 맨우측 사각형 x, y 좌표 차이
        plate_width = 0
        plate_height = 0
        #바로 오른쪽 사각형과의 x 좌표 차이를 구하는 변수 구해야함
        k = 0
        # k와 j의 x와 y 좌표 차이 (바로 맞닿은 사각형끼리의 차이 k와 j는 계속 변함)
        side_x = 0
        side_y = 0
        for j in range((i+1),len(rect_List)):
            dif_x = abs(rect_List[i].x2-rect_List[j].x1) # 문자 하나의 끝과 다음 사각형 문자 시작 사이의 x거리차이
            dif_y = abs(rect_List[i].y1-rect_List[j].y1)
            # 첫번째 합쳐지는 j 찾음 = k
            if(dif_x < 30 and dif_y < 5):
                rect_List[j].live = 0
                plate_width = abs(rect_List[i].x2-rect_List[j].x2) # i 끝과 j 끝의 x좌표 차이
                plate_height = dif_y
                k=j
                
            # 첫번째 이후 합쳐지는 사각형들을 찾음
            if(k != 0):
                side_x = abs(rect_List[k].x2 - rect_List[j].x1)
                side_y = abs(rect_List[k].y1 - rect_List[j].y1)
                if(side_x<30 and side_y < 5):
                    rect_List[j].live = 0
                    plate_width = plate_width + abs(rect_List[k].x2-rect_List[j].x2)
                    k=j
            #높이 더 높은쪽으로 맞춰줌
                if(rect_List[i].y1 > rect_List[k].y1):
                    rect_List[i].y1 = rect_List[k].y1
        rect_List[i].x2 = rect_List[i].x2 + plate_width

    # 리스트에서 live ==0 인 인덱스 제거
    rect_List2 = []
    for x in rect_List:
        if(x.live==1):
            rect_List2.append(x);
#가로 사각형 합치기 끝

#버블 정렬로 사각형들 위에서 아래순으로 정렬함(가장 위 사각형이 0)
    for i in range(len(rect_List2)):
        for j in range(0,len(rect_List2)-(i+1)):
            if (rect_List2[j].y1 > rect_List2[j+1].y1):
                temp_rect = rect_List2[j]
                rect_List2[j] = rect_List2[j+1]
                rect_List2[j+1] = temp_rect
            
# 세로 사각형 합치기
    for i in range(len(rect_List2)-1):
        # 사각형 i와 사각형 j의 가로, 세로 차이
        dif_x = 0
        dif_y = 0
        #맨위측 과 맨아래측 사각형 x, y 좌표 차이
        plate_width = 0
        plate_height = 0
        #바로 오른쪽 사각형과의 x 좌표 차이를 구하는 변수 구해야함
        k = 0
        # k와 j의 x와 y 좌표 차이 (바로 맞닿은 사각형끼리의 차이 k와 j는 계속 변함)
        side_x = 0
        side_y = 0
        for j in range((i+1),len(rect_List2)):
            dif_x = abs(rect_List2[i].x1-rect_List2[j].x1) #가로 합치는 부분과 다름, i와 j의 사각형 시작 x좌표 차이
            dif_y = abs(rect_List2[i].y2-rect_List2[j].y1) #가로 합치는 부분과 다름, i의 끝과 j의 시작 사이의 y좌표 차이
            # 첫번째 합쳐지는 j 찾음 = k
            if(dif_x < 50 and dif_y < 30):
                rect_List2[j].live = 0
                plate_width = abs(rect_List2[i].x2-rect_List2[j].x2)
                plate_height = abs(rect_List2[i].y2-rect_List2[j].y2) # i 끝과 j 끝의 y좌표 차이
                k=j
                
            # 첫번째 이후 합쳐지는 사각형들을 찾음
            if(k != 0):
                side_x = abs(rect_List2[k].x1 - rect_List2[j].x1)
                side_y = abs(rect_List2[k].y2 - rect_List2[j].y1)
                if(side_x<50 and side_y < 30):
                    rect_List2[j].live = 0
                    plate_width = plate_width + abs(rect_List2[k].x2-rect_List2[j].x2)
                    plate_height = plate_height + abs(rect_List2[k].y2-rect_List2[j].y2)
                    k=j
                #가로 더 긴쪽으로 맞춰줌
                if(rect_List2[i].x1 > rect_List2[k].x1):
                    rect_List2[i].x1 = rect_List2[k].x1
                if(rect_List2[i].x2 < rect_List2[k].x2):
                    rect_List2[i].x2 = rect_List2[k].x2
        rect_List2[i].y2 = rect_List2[i].y2 + plate_height
#세로 사각형 합치기 끝

    # 리스트에서 live ==0 인 인덱스 제거 제거
    rect_List3 = []
    for x in rect_List2:
        if(x.live==1):
            rect_List3.append(x);
    
    return rect_List3


# In[518]:


#텐서플로우에 전달할 이미지를 저장할 배열
image_List=[]

#rectangle 배열
rect_List = []

#이미지 변수에 저장
src = cv2.imread("testImage.png", cv2.IMREAD_UNCHANGED)
#drawContours 가 원본이미지를 변경하기에 이미지 복사
img1 = src.copy() #처음 Contours 그려짐
img2 = src.copy() #Rectangle Contours 그려짐

#CannyEdge
canny = change1(src)

#Contours 찾음
contours, hierachy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#그림에 Contours 그림
img1=cv2.drawContours(img1, contours, -1, (0,255,0),1)
#Contours를 사각형으로 만듬
for cnt in contours:
    
    # 정해진 크기가 아닌 사격형 Contours 그리지 않음
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    if (h<10) or (h>100) or (w>100):
        continue
    if (aspect_ratio>1.5) and(aspect_ratio<=0.2) :
        continue
    
    #rectangle 좌표들 배열에 저장
    rect_List.append(rectang(x, y, x+w, y+h))

    #######중심함수
rect_List=combineRectang(rect_List)

for o3 in range(len(rect_List)):
    img2 = cv2.rectangle(img2,(rect_List[o3].x1, rect_List[o3].y1),(rect_List[o3].x2, rect_List[o3].y2),(0,255,0),1)
    
    #배열에 텐서플로우에 전달할 이미지 저장
    dst = src.copy()
    dst = src[rect_List[o3].y1:rect_List[o3].y2, rect_List[o3].x1:rect_List[o3].x2]
    dst_gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    dst_ret, dst_gray = cv2.threshold(dst_gray, 127,255,cv2.THRESH_BINARY)

    print('세로',rect_List[o3].y2-rect_List[o3].y1, '가로', rect_List[o3].x2-rect_List[o3].x1)
    image_List.append(dst_gray)

#사각형으로 변현한 Contours 출력
cv2.imshow("img2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

