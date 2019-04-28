
# coding: utf-8

# In[15]:


import cv2
import numpy as np
import os

#사각형
class rectang:
    x1=0
    y1=0
    x2=0
    y2=0
    live=1
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
#     cv2.imshow("canny",canny)
    return canny


# In[16]:


#폴더내의 이미지 배열로 불러옴
i=1
j=1
path_dir = 'C:\st2/'
file_list = os.listdir(path_dir)
# 폴더에 있는 이미지를 전부 잘라서 저장함
number = 1
for cutImage in file_list:
    #텐서플로우에 전달할 이미지를 저장할 배열
    image_List=[]
    image_List2=[]
    #rectangle 배열
    rect_List = []
    rect_List2=[]
    print(cutImage)
    #이미지 변수에 저장        
    path = 'C:\st2/'
    src = cv2.imread(path+cutImage, cv2.IMREAD_UNCHANGED)
    
    #drawContours 가 원본이미지를 변경하기에 이미지 복사
    img1 = src.copy() #처음 Contours 그려짐
    img2 = src.copy() #Rectangle Contours 그려짐
    img3 = src.copy() #정리후 Rectangle Contours 그려짐

    #CannyEdge
    canny = change1(src)

    #Contours 찾음
    contours, hierachy = cv2.findContours(canny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #그림에 Contours 그림
    cv2.drawContours(img1, contours, -1, (0,255,0),1)
    #Contours를 사각형으로 만듬
    for cnt in contours:
    
        #크기 작은 사격형 Contours 그리지 않음
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        if (w<50) or (h<30):
            continue
        #rectangle 좌표들 배열에 저장
        rect_List.append(rectang(x, y, x+w, y+h))

    for r1 in rect_List:
        img2 = cv2.rectangle(img2,(r1.x1, r1.y1),(r1.x2, r1.y2),(0,255,0),1)
    cv2.imshow("hh", img2)

    check=0
    #사각형 내부의 사각형 제거. 가로,세로 좌표가 다른 사각형 내부에 포함되면 그려지지않게함
    for r1 in rect_List:
        check=1
        if(r1.live==1):
            for r2 in rect_List:
                if(r2.live==1 and check==1):
                    if (((r1.x1 < r2.x1) and (r2.x1 < r1.x2)) or ((r1.x1 < r2.x2)and (r2.x2< r1.x2))):
                        if (((r1.y1 < r2.y1) and (r2.y1 < r1.y2)) or ((r1.y1 < r2.y2)and (r2.y2< r1.y2))):
                            print("combine")
                            r1.live=0
                            r2.live=0
                            check=0;
                            rect_List2.append(rectang(min(r1.x1, r2.x1),min(r1.y1, r2.y1), max(r1.x2, r2.x2), max(r1.y2, r2.y2)))
                            continue
            if(r1.live==1):
                rect_List2.append(rectang(r1.x1, r1.y1, r1.x2, r1.y2))
        #해당 될시 그리는부분 스킵


    for r1 in rect_List2:

        img3 = cv2.rectangle(img3,(r1.x1, r1.y1),(r1.x2, r1.y2),(0,255,0),1)
    
        #배열에 텐서플로우에 전달할 이미지 저장
        dst = src.copy()
        dst = src[r1.y1:r1.y2, r1.x1:r1.x2]
        dst_gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
        dst_ret, dst_gray = cv2.threshold(dst_gray, 127,255,cv2.THRESH_BINARY)
        dst_laplacian = cv2.Laplacian(dst_gray, cv2.CV_8U)
        dst_laplacian = cv2.resize(dst_laplacian, dsize=(100, 100), interpolation=cv2.INTER_AREA)
        image_List.append(dst_gray)
        image_List2.append(dst_laplacian)


    cv2.imshow("gg", img3)
    '''
    for passImage in image_List:
        name = "image"
        #자른 이미지 저장@@
        path='../text_recog_module/negative/'+name+str(i)
        cv2.imwrite(path+'.jpg', passImage)
        i=i+1
    '''
    '''
    for passImage in image_List4:
        name = "L_image"
        # 자른 이미지 저장@@
        path = 'image/trainImage2/' + name + str(i)
        cv2.imwrite(path + '.jpg', passImage)
        j = j + 1
    '''
    cv2.waitKey(0)
    cv2.destroyAllWindows()
