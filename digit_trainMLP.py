import matplotlib.image as mimage
from sklearn import datasets,svm,metrics
from sklearn.model_selection import cross_val_score,train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.externals import joblib


data=np.zeros((10*60,240))
target=np.zeros((10*60))
count=0

for i in range(0,10):#will traverse folder
    for j in range(0,60):#will traverse sample image
        path=('./Train/%d/%d.png'%(i,j))
        im=mimage.imread(path)
        v1=im.mean(axis=1)
        v1=v1.reshape(1,-1)
        data[count,:]=v1
        target[count]=i
        count=count+1

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.2)

    
neural_model=MLPClassifier(100,max_iter=2000)#hidden layer = 100
#training of the network
neural_model=neural_model.fit(train_data,train_target)
#testing data
joblib.dump(neural_model,'train_digits6.pkl')
print('Trained model saved')


#cross_validation for 10 k-folds for iris datasets

#performance
#acc=metrics.accuracy_score(test_target,output)
#print("Accuracy::",acc*100)
#conf_mat=metrics.confusion_matrix(test_target,output)
#print("Confusion Matrix::",conf_mat)
