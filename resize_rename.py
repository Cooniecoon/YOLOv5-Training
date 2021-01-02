import cv2
import os

### 원본 폴더 경로 작성
path = '../dataset/'

### 리사이즈 이미지를 저장할 폴더 경로 작성
save = '../data_1212/'

pixel_X=640
pixel_Y=640

#! "image_rename"
File_Name="person_1212_"



def changeName(path, cName):
    i=0
    for filename in os.listdir(path):
        print(path+filename, '=>', path+str(cName)+str(i)+'.jpg')
        os.rename(path+filename, path+str(cName)+str(i)+'.jpg')
        i +=1

changeName(path,File_Name)

img_list=os.listdir(path)
# print(img_list)

for i in range(len(os.listdir(path))):
    img=cv2.imread(path+img_list[i],cv2.IMREAD_COLOR)
    
    # 1. general or big
    #dst = cv2.resize(img, dsize=(pixel_X, pixel_Y), interpolation=cv2.INTER_LINEAR)

    # 2. small
    dst = cv2.resize(img, dsize=(pixel_X, pixel_Y), interpolation=cv2.INTER_AREA)

    cv2.imwrite(save+img_list[i],dst)
