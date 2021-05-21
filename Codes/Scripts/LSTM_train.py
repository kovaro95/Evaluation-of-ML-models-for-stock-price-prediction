import random as python_random

import numpy as np
import pandas as pd
import sys
import datetime

import psycopg2

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

from keras.models import Model
from keras.models import Sequential
from keras import optimizers
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import json

import scipy.optimize as optimize

class Dataset():
    def __init__(self, data,look_forw=1,look_back=60,test_size=0.2):    
        self.split=int(len(data)*(1-test_size))
        self.Scaler=None
        self.look_forw = look_forw
        self.look_back=look_back
        
        self.Train = data.iloc[:self.split].values
        self.Test = data.iloc[self.split-self.look_back:].values
        
        #SCALING
        self.Scaler=MinMaxScaler(feature_range = (-1, 1))
        self.Scaler=self.Scaler.fit(self.Train)
    
        self.Train=self.Scaler.transform(self.Train)
        self.Test=self.Scaler.transform(self.Test)
        ##
        
        self.X_train_seq=[]
        self.y_train_seq=[]
        for i in range(self.look_back,len(self.Train)-self.look_forw+1):
            self.X_train_seq.append(self.Train[i-self.look_back:i,:])                
            self.y_train_seq.append(self.Train[i+self.look_forw-1])

        self.X_train_seq=np.array(self.X_train_seq).astype('float32')
        self.y_train_seq=np.array(self.y_train_seq,dtype='object').astype('float32')

        self.X_train_seq=self.X_train_seq.reshape(self.X_train_seq.shape[0],self.X_train_seq.shape[1],self.X_train_seq.shape[2])

        self.X_test_seq=[]
        for i in range(self.look_back,len(self.Test)):
                self.X_test_seq.append(self.Test[i-self.look_back:i,:])
                
        self.X_test_seq=np.asarray(self.X_test_seq).astype('float32')
        self.X_test_seq=self.X_test_seq.reshape(self.X_test_seq.shape[0],self.X_test_seq.shape[1],self.X_test_seq.shape[2])

        print(self.__repr__())

    def __repr__(self):
        return '\n'.join([
        f'Original train and test{self.Train.shape,self.Test.shape}',
        f'X train size {self.X_train_seq.shape}',
        f'Y train size: {self.y_train_seq.shape}',
        f'X test size: {self.X_test_seq.shape}']) 

def ensemble_loss(weights):
    weighted_preds=np.dot(weights,np.squeeze(lstm_prediction))
    return RMSE(true_stock,weighted_preds)

def MAPE(y_hat,y_pred):
    mape = np.mean(np.abs((y_hat - y_pred)/y_hat))*100
    return mape

def RMSE(y_hat,y_pred):
    MSE = np.square(np.subtract(y_hat,y_pred)).mean() 
    return (MSE**(1/2))
 
def Start_training(SELECTED_STOCK):
    global lstm_prediction
    global true_stock
    phys_dev=tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(phys_dev[0],True)    
    keras.backend.clear_session()

    #PARAMETERS
    EPOCHS=1000
    BATCH_SIZE=256
    RANDOMSEED=123

    np.random.seed(RANDOMSEED)
    python_random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)
        
    sql=f'SELECT {SELECTED_STOCK} FROM stockprices'
    connection = psycopg2.connect(user="postgres",
        password="adminpw",
        host="127.0.0.1",
        port="5432",
        database="Thesis")
    
    stock = pd.read_sql_query(sql,connection)
    connection.close()

    if stock.shape[0]==0:
        sys.exit(f'Stock data failed to download for {SELECTED_STOCK}')

    data=Dataset(stock,look_back=60,look_forw=1)
    true_stock=stock.iloc[data.split:].values
    input_shape=(data.look_back,data.X_train_seq.shape[2])
    
    _mape=tf.keras.metrics.MeanAbsolutePercentageError()
    _rmse=tf.keras.metrics.RootMeanSquaredError()
    
    #CALLBACKS
    es=EarlyStopping(monitor='val_loss',min_delta=1e-3,mode="min",patience=10,verbose=1)
    rlr=ReduceLROnPlateau(monitor='val_loss',min_delta=1e-2,factor=0.75,cooldown=5,mode='min',patience=3,verbose=1)
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)   

    LSTM_Models={'Name':[],'Model':[],'History':[],'Preds':[]}   
    
    default_lstm = Sequential([
        layers.LSTM(units=50, return_sequences=True,input_shape=input_shape),
        layers.LSTM(units=50, return_sequences=False),
        layers.Dense(units=data.y_train_seq.shape[1])
    ]) 
     
    base_lstm = Sequential([
    layers.LSTM(units=50, return_sequences=True,input_shape=input_shape,activation='linear'),
    layers.Dropout(0.1),
    layers.LSTM(units=50, return_sequences=False),
    layers.Dropout(0.1),
    layers.Dense(units=data.y_train_seq.shape[1])
    ]) 
    
    gelu_lstm = Sequential([
    layers.LSTM(units=50, return_sequences=True,input_shape=input_shape,activation='gelu'),
    layers.Dropout(0.8),
    layers.LSTM(units=50, return_sequences=False),
    layers.Dropout(0.8),
    layers.Dense(units=data.y_train_seq.shape[1])
    ]) 
    
    swish_lstm = Sequential([
    layers.LSTM(units=50, return_sequences=True,input_shape=input_shape,activation='swish'),
    layers.Dropout(0.1),
    layers.LSTM(units=50, return_sequences=False),
    layers.Dropout(0.1),
    layers.Dense(units=data.y_train_seq.shape[1],activation='relu')
    ]) 
    
    default_lstm.compile(optimizer=optimizer,loss=['mse'],metrics=[_mape])
    LSTM_Models['Name'].append('Default LSTM')
    LSTM_Models['Model'].append(default_lstm)    
    
    base_lstm.compile(optimizer=optimizer,loss=['mse'],metrics=[_mape])
    LSTM_Models['Name'].append('Base LSTM')
    LSTM_Models['Model'].append(base_lstm)
    
    gelu_lstm.compile(optimizer=optimizer,loss=['mse'],metrics=[_mape])
    LSTM_Models['Name'].append('Gelu LSTM')
    LSTM_Models['Model'].append(gelu_lstm)
    
    swish_lstm.compile(optimizer=optimizer,loss=['mse'],metrics=[_mape])
    LSTM_Models['Name'].append('Swish LSTM')
    LSTM_Models['Model'].append(swish_lstm)
    
    LSTM_write={'model_name':'','weights':[],'r_squared':0,'rmse':0,'mape':0}    
    LSTM_write['model_name']=LSTM_Models['Name']
    
    i=0
    for model in LSTM_Models['Model']:
        MODEL_NAME=LSTM_Models["Name"][i]
        log_dir = f'../../Data/Logs/{MODEL_NAME}/{SELECTED_STOCK}'
        tboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        history=model.fit(data.X_train_seq,data.y_train_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=0, callbacks=[es,rlr,tboard])
        model.save(f'../../Models/{MODEL_NAME}/{SELECTED_STOCK}')
        preds_scaled=model.predict(data.X_test_seq)
        preds=data.Scaler.inverse_transform(preds_scaled)
        LSTM_Models['Preds'].append(preds.reshape(-1,1))
        LSTM_Models['History'].append(history)
        sys.stdout.write(f'{MODEL_NAME} is done')  # same as print
        sys.stdout.flush()
        i+=1
        
    lstm_prediction=np.array(LSTM_Models['Preds'])
    
    opt_weights = optimize.minimize(ensemble_loss,
                                [1/len(LSTM_Models.keys())] * len(LSTM_Models.keys()),
                                constraints=({'type': 'eq','fun': lambda w: 1-sum(w)}),
                                method= 'SLSQP', 
                                bounds=[(0.0, 1.0)] * len(LSTM_Models.keys()),
                                options = {'ftol':1e-10},
                            )['x']
    LSTM_write['weights']=(np.squeeze(opt_weights)).tolist()
    LSTM_write['r_squared']=[r2_score(true_stock,s_pred) for s_pred in LSTM_Models['Preds']]
    LSTM_write['rmse']=[RMSE(true_stock,s_pred) for s_pred in LSTM_Models['Preds']]
    LSTM_write['mape']=[MAPE(true_stock,s_pred) for s_pred in LSTM_Models['Preds']]
    
    with open(f'../../Data/Models/{SELECTED_STOCK}.json','w+') as f:
        json.dump(LSTM_write, f)
    
    np.savetxt(f'../../Data/Weights/{SELECTED_STOCK}_opt_weight.txt',opt_weights)
    
if __name__== "__main__":
    Start_training("MSFT")   