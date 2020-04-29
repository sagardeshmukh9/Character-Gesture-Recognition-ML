#CODE FOR CREATING AND STORING SAMPLES
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from scipy import ndimage
from collections import deque
import argparse
from sklearn.externals import joblib
from sklearn import svm,datasets,metrics
from PIL import Image

d=40
def nothing(x):
  pass
cap = cv2.VideoCapture(0)
cv2.namedWindow("T")
cv2.createTrackbar("LH","T",0,179,nothing)
cv2.createTrackbar("LS","T",0,255,nothing)
cv2.createTrackbar("LV","T",0,255,nothing)
cv2.createTrackbar("UH","T",0,179,nothing)
cv2.createTrackbar("US","T",0,255,nothing)
cv2.createTrackbar("UV","T",0,255,nothing)


ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64,help="max buffer size")
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"])

while True:
    ret, img = cap.read()
    wimg  = cv2.imread("alpha000.jpg")
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3),np.uint8)
    
    #region of interest
    roi=img[80:320,380:620]
    wroi=wimg[80:320,380:620]
    cv2.rectangle(img,(380,80),(620,320),(0,255,0),0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #Skin color range in hsv
    LH=cv2.getTrackbarPos("LH","T")
    LS=cv2.getTrackbarPos("LS","T")
    LV=cv2.getTrackbarPos("LV","T")
    UH=cv2.getTrackbarPos("UH","T")
    US=cv2.getTrackbarPos("US","T")
    UV=cv2.getTrackbarPos("UV","T")
    
    lower_tone = np.array([LH,LS,LV], dtype=np.uint8)
    upper_tone = np.array([UH,US,UV], dtype=np.uint8)

    #extract object of hsv range from roi
    mask=cv2.inRange(hsv,lower_tone,upper_tone)
    mask = cv2.dilate(mask,kernel,iterations = 4)
    mask = cv2.GaussianBlur(mask,(5,5),100)

    mask=cv2.flip(mask,1)
    mask=cv2.transpose(mask,mask)
    mask=cv2.flip(mask,0)

    #finding centre of object
    ret,thresh_img = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    arr=ndimage.distance_transform_edt(thresh_img)
    #storing the pixel with max value
    ind = np.unravel_index(np.argmax(arr, axis=None), arr.shape)
    pts.append(ind)
    #generating a line to depict the gesture
    for i in range(1,len(pts)):
        cv2.line(roi, pts[i - 1], pts[i], (0, 255, 0),3)
        cv2.line(wroi, pts[i - 1], pts[i], (255, 255, 255),3)
    cv2.circle(roi,(ind[0],ind[1]),3,[255,0,0],-1)
    
    wroigray=cv2.cvtColor(wroi,cv2.COLOR_BGR2GRAY)
    wroigray = cv2.dilate(wroigray,kernel,iterations = 4)
    wroigray = cv2.GaussianBlur(wroigray,(5,5),100)
    #fetching attributes
    v1=wroigray.mean(axis=1)
    v1=v1.reshape(1,-1)
    
    cv2.imshow('gwroi',wroigray)
    cv2.imshow('mask',mask)
    cv2.imshow('video',img)
   
    name='./Train/9/'+str(d)+'.png'
    k = cv2.waitKey(10)
    if k == 27: # press 'ESC' to quit
        cv2.imwrite(name,wroigray)
        print("sample "+str(d+1)+" saved")
        d=d+1
        
    

cap.release()
cv2.destroyAllWindows()
plt.show()
