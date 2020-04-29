import cv2
import numpy as np
import pandas as pd
import matplotlib.image as mimage
import matplotlib.pyplot as plt
from sklearn import svm,metrics,datasets
from sklearn.externals import joblib

train_data=np.zeros((61200,240))
train_target=np.zeros((61200))
count=0

for i in range(0,10):
    for j in range(0,60):
        path=(r'./Train/%d/%d.png'%(i,j))
        im=mimage.imread(path)
        
        v1=im.mean(axis=1)
        v1=v1.reshape(1,-1)
        train_data[count,:]=v1
        train_target[count]=i
        count=count+1
        

svm_model=svm.SVC(kernel='linear')
#training of fingerprint train datasets
svm_model=svm_model.fit(train_data,train_target)
#save the trained model
joblib.dump(svm_model,'train_digits5.pkl')
print('Trained model saved')
