# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 02:40:50 2017

@author: Mahsa
"""

import numpy as np
from numpy.random import permutation
import pandas as pd
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected,flatten
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


nummberclass = 10
filter_size_1 = 6
filter_Size_2 = 12
filter_size_3 = 24
fullyconnected_size = 200

train_Path = r'D:\digit\train.csv'
test_Path = r'D:\digit\test.csv'
  
#Split arrays or matrices into random train and test subsets
#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

def split_matrices_into_random_train_test_subsets(train_Path):
    train = pd.read_csv(train_Path)
    train = np.array(train)
    train = permutation(train)
    X = train[:,1:785].astype(np.float32) #feature
    y = train[:,0].astype(np.float32) #label
    return train_test_split(X, y, test_size=0.33, random_state=42)

def reshape_data(Data,Labels):
    Data = Data.reshape(-1,28,28,1).astype(np.float32)
    Labels = (np.arange(nummberclass) == Labels[:,None]).astype(np.float32)
    return Data,Labels
  
def Convolution_NN(input_size,arg):
    input_layer = input_data(shape=[None,input_size,input_size,arg],name='input_layer')
    conv1 = conv_2d(input_layer,nb_filter=filter_size_1,filter_size=6,strides=1,activation='relu',regularizer='L2')
    #conv1 = max_pool_2d(conv1,2)
    
    conv2 = conv_2d(conv1,nb_filter=filter_Size_2,filter_size=5,strides=2,activation='relu',regularizer='L2')
    #conv2 = max_pool_2d(conv2,2)
    conv3 = conv_2d(conv2,nb_filter=filter_size_3,filter_size=4,strides=2,activation='relu',regularizer='L2')
    
    full_layer1 = fully_connected(flatten(conv3),fullyconnected_size,activation='relu',regularizer='L2')
    full_layer1 = dropout(full_layer1,0.75)
    
    out_layer = fully_connected(full_layer1,10,activation='softmax')
    
        
    sgd = tflearn.SGD(learning_rate=0.1,lr_decay=0.096,decay_step=100)
    
    top_k = tflearn.metrics.top_k(3)
    
    network = regression(out_layer,optimizer=sgd,metric=top_k,loss='categorical_crossentropy')
    return tflearn.DNN(network,tensorboard_dir='tf_CNN_board',tensorboard_verbose=3)
 

X_train, X_test, y_train, y_test = split_matrices_into_random_train_test_subsets(train_Path)


X_train,y_train = reshape_data(X_train,y_train)
X_test,y_test = reshape_data(X_test,y_test)

test_x = np.array(pd.read_csv(test_Path))
test_x = test_x.reshape(-1,28,28,1)


model = Convolution_NN(input_size=28,arg=1)

model.fit(X_train, y_train, batch_size=128, validation_set=(X_test,y_test), n_epoch=20, show_metric=True)

P = model.predict(test_x)

index = [i for i in range(1,len(P)+1)]
result = []
for i in range(len(P)):
    result.append(np.argmax(P[i]).astype(np.int))

res = pd.DataFrame({'ImageId':index,'Label':result})
res.to_csv("sample_submission.csv",index=False)
