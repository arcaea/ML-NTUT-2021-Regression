#import
import pandas as pd #讀取csv
import numpy as np
from sklearn.preprocessing import * #正規化作用
from keras.models import Sequential #導入模型
from keras.layers import * #層數
from keras.callbacks import *
from pathlib import Path #路徑
from tensorflow import keras
from tensorflow.keras import layers

#與google drive連接
DataRoot='/content/drive/MyDrive/Colab Notebooks/NTUT_MachineLearning/HW1/Data' #資料位置
print(os.listdir(DataRoot))

Target='price'
col=['sqft_living','grade','sqft_above','sqft_living15','bathrooms','view',
     'sqft_basement','lat','bedrooms','waterfront','floors','yr_renovated',
     'sqft_lot','sqft_lot15','yr_built','condition','long','sale_yr']#所有正相關
#train data
data1=pd.read_csv(f'{DataRoot}/train-v3.csv')
Xtrain=data1[col]
Ytrain=data1[Target]

#valid data
data2=pd.read_csv(f'{DataRoot}/valid-v3.csv')
Xvalid=data2[col]
Yvalid=data2[Target]

#test data
data3=pd.read_csv(f'{DataRoot}/test-v3.csv')
Xtest=data3[col]

data1.corr()[['price']].sort_values(by='price',ascending=False)

#正規化
sc=StandardScaler().fit(Xtrain)
Xtrain=scale(Xtrain)
Xvalid_RE=sc.transform(Xvalid)
Xtest=sc.transform(Xtest)

model=Sequential()
model.add(Dense(32,input_dim=Xtrain.shape[1],kernel_initializer='he_normal',activation='relu'))
model.add(Dense(64,input_dim=32,kernel_initializer='he_normal',activation='relu'))
model.add(Dense(128,input_dim=64,kernel_initializer='he_normal',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,input_dim=64,kernel_initializer='he_normal',activation='relu'))
model.add(Dense(32,input_dim=32,kernel_initializer='he_normal',activation='relu'))
model.add(Dense(1,activation='linear'))

model.summary()

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='MAE',optimizer=opt)
epochs=200 #train次數
batchSize=16 #抽的data數量-分批亂數抽取

fileName=str(epochs)+"_"+str(batchSize)
call=TensorBoard(log_dir='logs/'+fileName,histogram_freq=0)
history=model.fit(Xtrain,Ytrain,batch_size=batchSize,epochs=epochs,verbose=1,validation_data=(Xvalid_RE,Yvalid),callbacks=[call])
