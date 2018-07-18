 # Prediction of Building Damage using nueral network
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('train.csv') 
X = data.iloc[:,3:14].values
y = data.iloc[:,2:3].values
from sklearn.preprocessing import Imputer
imputer =Imputer(missing_values = 'NaN' ,strategy ='mean',axis =0)
imputer = imputer.fit(X[:,3:14])
X[:,3:14]=imputer.transform(X[:,3:14])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_y=LabelEncoder()
y[:,0]=labelencoder_y.fit_transform(y[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
y=onehotencoder.fit_transform(y).toarray() 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.5,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
 
import keras
from keras.models import Sequential
from keras.layers import Dense
 
classifier  = Sequential()
classifier.add(Dense(  6,kernel_initializer='uniform', activation = 'relu',input_dim = 11))
classifier.add(Dense(  6,kernel_initializer ='uniform', activation='relu'))
classifier.add(Dense( 5,kernel_initializer = 'uniform',activation = 'sigmoid'))
classifier.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train , y_train , batch_size=10 , epochs = 25)
 
y_pred = classifier.predict(X_test)
y_new=np.zeros(315881) 
for i in range(0,315881):
        k=0
        for j in range(1,5):
            pivot=y_pred[i][k]
            if y_pred[i][j] > pivot:
                pivot=y_pred[i][j]
                k=j
        y_new[i]=k
            
y_pred1=np.zeros(315881) 
for i in range(0,315881):
        k=0
        for j in range(1,5):
            pivot=y_test[i][k]
            if y_test[i][j] > pivot:
                pivot=y_test[i][j]
                k=j
        y_pred1[i]=k
        
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_new,y_pred1)
