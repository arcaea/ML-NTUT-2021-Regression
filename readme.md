環境要求
=======
matplot==0.1.9 \
numpy==1.18.5 \
pandas==1.0.5 \
Keras-Preprocessing==1.1.2 \
seaborn==0.10.1

流程
=======
1.用train-v3.csv及valid-v3.csv訓練模型\
2.將test-v3.csv中每一筆方屋參數，輸入訓練好的模型去預測房價\
3.將預測解果上傳kaggle\
4.嘗試改進模型

資料分析
=======
```python
#讀入資料
#-------------------------------------------------------------------------------------
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

#資料預處理
#-------------------------------------------------------------------------------------
data1.corr()[['price']].sort_values(by='price',ascending=False)

#資料正規化=(原始資料-平均值)/標準差
#-------------------------------------------------------------------------------------
sc=StandardScaler().fit(Xtrain)
Xtrain=scale(Xtrain)
Xvalid_RE=sc.transform(Xvalid)
Xtest=sc.transform(Xtest)
```

做法
=======
![image](https://github.com/MachineLearningNTUT/regression-NTUB110002016/blob/main/HW1/Picture/model.JPG?raw=true)

程式寫法
=======
```python
#import
#-------------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------------
DataRoot='/content/drive/MyDrive/Colab Notebooks/NTUT_MachineLearning/HW1/Data' #資料位置
print(os.listdir(DataRoot))

#讀入資料
#-------------------------------------------------------------------------------------
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

#資料預處理
#-------------------------------------------------------------------------------------
data1.corr()[['price']].sort_values(by='price',ascending=False)

#資料正規化=(原始資料-平均值)/標準差
#-------------------------------------------------------------------------------------
sc=StandardScaler().fit(Xtrain)
Xtrain=scale(Xtrain)
Xvalid_RE=sc.transform(Xvalid)
Xtest=sc.transform(Xtest)

#建立模型
#-------------------------------------------------------------------------------------
model=Sequential()
model.add(Dense(32,input_dim=Xtrain.shape[1],kernel_initializer='he_normal',activation='relu'))
model.add(Dense(64,input_dim=32,kernel_initializer='he_normal',activation='relu'))
model.add(Dense(128,input_dim=64,kernel_initializer='he_normal',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,input_dim=64,kernel_initializer='he_normal',activation='relu'))
model.add(Dense(32,input_dim=32,kernel_initializer='he_normal',activation='relu'))
model.add(Dense(1,activation='linear'))

#訓練資料
#-------------------------------------------------------------------------------------
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='MAE',optimizer=opt)
epochs=200 #train次數
batchSize=16 #抽的data數量-分批亂數抽取

fileName=str(epochs)+"_"+str(batchSize)
call=TensorBoard(log_dir='logs/'+fileName,histogram_freq=0)
history=model.fit(Xtrain,Ytrain,batch_size=batchSize,epochs=epochs,verbose=1,validation_data=(Xvalid_RE,Yvalid),callbacks=[call])

#利用模型預測test檔並儲存
#-------------------------------------------------------------------------------------
model.save('h5/'+fileName+'.h5')

testPR=model.predict(Xtest)
print(testPR)

with open('TEST110002016_ver18.csv','w') as f:
  f.write('id,price\n')
  for i in range(len(testPR)):
    f.write(str(i+1)+","+str(float(testPR[i]))+"\n")
```

實際與預測差距大\
原因:\
1.資料異常值\
2.訓練特徵不足或過多\
3.非最佳模型架構


檢討與改進
=======
1.改變模型\
2.微調參數，例如:learning rate或batch size

備註
=======
Picture、h5、test、train、validation資料夾皆為執行ntut_hw1_ver18.py結果後的產出或截圖
