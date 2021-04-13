#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIMAT UNIDAD MONTERREY
MAESTRIA EN COMPUTO ESTADISTICO
TERCER SEMESTRE, MATERIA TEMAS SELECTOS DE COMPUTO
VICTOR MANUEL GOMEZ ESPINOSA
"""
#librerias
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


#Clase que realiza la imputacion de datos faltantes en un DataFrame por distintos metodos
class MissingValueImputation():
    
    def __init__(self, method = 'list_wise', k = 40, learning_rate=0.6, n_estimators=300, subsample=0.9, gamma=100, reg_lambda=100,neur_Hid_layer=[50,50],epoch=100):
        self.method = method
        self.k = k # k vecinos mas cercanos
        self.learning_rate=learning_rate
        self.n_estimators=n_estimators
        self.subsample=subsample
        self.gamma=gamma
        self.reg_lambda=reg_lambda
        self.neur_Hid_layer=neur_Hid_layer #numero de neuronas en cada paca oculta
        self.epoch=epoch
        
    def df_clean(self,df): #libera memoria
        df=pd.DataFrame()
        del df
        return self
        
    def __reference(self, df): #obtiene informacion del data frame
        self.types_df=df.dtypes #tipo de datos de las columnas
        
        df_na=pd.isna(df) #celdas que contienen datos faltantes NaN
        na_rows=df_na.sum(1) #renglones con datos faltantes
        na_features=df_na.sum(0)>0 #columnas con datos faltantes
        
        self.ind_columns_na=np.where(na_features>0)[0] #indices de las columnas con datos faltantes
        self.ind_rows_complete=np.where(na_rows==0)[0] #indices de las filas con datos completos
        
        df_complete=df[na_rows==0] #dataframe sin valores faltantes
        
        
        n_com=df_complete.shape[0]
        if n_com<self.k:
            self.k=n_com
        
        
        self.df_clean(df_complete)
        return self
    
    def __mean_mode_model(self, df):
        columns=self.ind_columns_na
        
        centroid_incomplete=[] #donde se guardaán los valores
        
        for column in columns: #recorre las columnas con datos faltantes
            col_type=self.types_df[column]
            
            if col_type=='float64' or col_type=='int64': #si es de tipo numérico
                center=np.nanmean(df.iloc[:,column].values) #media
            else: #categorico
                d_rows_complete=np.where(pd.notna(df.iloc[:,column]))[0]    #indices de valores completos en la columna de interes (y)
                center=stats.mode(df.iloc[d_rows_complete,column].values,nan_policy="omit")[0][0] #moda
            centroid_incomplete.append(center)
            
        self.centroid_incomplete = centroid_incomplete
        return self
    
    def __mean_mode_imputation(self, df):
        centroid=self.centroid_incomplete 
        columns=self.ind_columns_na
        
        
        for column, value in zip(columns, centroid): #recorre las columnas con datos faltantes y los llena con los valores
            df.iloc[:,column]=df.iloc[:,column].fillna(value)
            
        
        return df
    
    def __k_nearest_neighbors_imputation(self, df):
        columns=self.ind_columns_na #columnas con datos faltantes
        indx_rows_complete=self.ind_rows_complete #indices de valores completos
        
        
        
        gbm_param_grid={'n_neighbors':np.random.randint(2,self.k, size=10)}
        
        
        
        
        df2=df.copy() #crea una copia del original y si hay datos faltantes los llena con el metodo de la media o moda
        self.method='mean_mode'
        df2=self.fit_transform(df2)
        self.method='knn'
        
        
        for column in columns: #recorre las columnas con datos faltantes
            col_type=self.types_df[column] #tipo de columna
            d_rows_na=np.where(pd.isna(df.iloc[:,column]))[0]    #indices de valores faltantes en la columna de interes (y)
            
            
            df3=df2.drop([column],axis=1) #elimina la columna a evaluar (y) del df completo
            df3=pd.get_dummies(df3, drop_first=True) #elimina categoricas y crea variables dummies
            
            df_complete=df3.iloc[indx_rows_complete,:] #selecciona los ejemplos completos
            df_incomplete=df3.iloc[d_rows_na,:] #selecciona los ejemplos vacios de la columna y
            
            
            
            
            X_train=df_complete.values
            X=df_incomplete.values
            
            y_train=df.iloc[indx_rows_complete,column].values
            
            if col_type=='float64' or col_type=='int64': #si es de tipo numérico
                pipe = Pipeline([('scaler', StandardScaler()), ('rndsearch',RandomizedSearchCV(estimator=KNeighborsRegressor(),param_distributions=gbm_param_grid,scoring='neg_mean_absolute_error',cv=3,return_train_score=True,n_jobs=-1))]) #pipe regresion
                y=pipe.fit(X_train, y_train).predict(X) #regresion
            else: #categorico
                pipe = Pipeline([('scaler', StandardScaler()), ('rndsearch',RandomizedSearchCV(estimator=KNeighborsClassifier(),param_distributions=gbm_param_grid,scoring='accuracy',cv=3,return_train_score=True,n_jobs=-1))]) #pipe class
                y=pipe.fit(X_train, y_train).predict(X) #clasificacion
            
            df.iloc[d_rows_na,column]=y
            
            
            del X_train, X, y_train, y
            self.df_clean(df_incomplete)
            self.df_clean(df_complete)
            self.df_clean(df3)
            self.df_clean(df2)
        return df
    
    def __trees_imputation(self, df):
        columns=self.ind_columns_na #columnas con datos faltantes
        indx_rows_complete=self.ind_rows_complete #indices de valores completos
        
        gbm_param_grid={'learning_rate':[0.1,0.3,self.learning_rate],
               'n_estimators':[50,150,self.n_estimators],
               'subsample':[self.subsample],
               'gamma':[0,10,self.gamma],
               'lambda':[1,10,self.reg_lambda]}
        
        
        df2=df.copy() #crea una copia del original y si hay datos faltantes los llena con el metodo de la media o moda
        
        
        
        for column in columns: #recorre las columnas con datos faltantes
            col_type=self.types_df[column] #tipo de columna
            d_rows_na=np.where(pd.isna(df.iloc[:,column]))[0]    #indices de valores faltantes en la columna de interes (y)
            d_rows_notna=np.where(pd.notna(df.iloc[:,column]))[0]    #indices sin valores faltantes en la columna de interes (y)
            num_classes = np.shape(df.iloc[indx_rows_complete,column].unique())[0] #numero de clases si es categorica
            
            df3=df2.drop([column],axis=1) #elimina la columna a evaluar (y) del df completo
            df3=pd.get_dummies(df3, drop_first=True) #elimina categoricas y crea variables dummies
            
            
            
            
            df_complete=df3.iloc[d_rows_notna,:] #selecciona los ejemplos completos
            df_incomplete=df3.iloc[d_rows_na,:] #selecciona los ejemplos vacios de la columna y
            
            
            
            
            X_train=df_complete.values
            X_pred=df_incomplete.values
            
            y_train=df.iloc[d_rows_notna,column].values
            
            if col_type=='float64' or col_type=='int64': #si es de tipo numérico
                gbm=xgb.XGBRegressor()
                rnd=RandomizedSearchCV(estimator=gbm,param_distributions=gbm_param_grid,scoring='neg_mean_absolute_error',cv=3,return_train_score=True,n_jobs=-1)
                rnd.fit(X_train,y_train)
                y=rnd.predict(X_pred) #regresion
            else: #categorico
                if num_classes ==2:
                    gbm=xgb.XGBClassifier()
                    rnd=RandomizedSearchCV(estimator=gbm,param_distributions=gbm_param_grid,scoring='accuracy',cv=3,return_train_score=True,n_jobs=-1)
                    rnd.fit(X_train,y_train)
                    y=rnd.predict(X_pred) #clasificacion
                else:
                    gbm=xgb.XGBClassifier(objective='multi:softmax')
                    rnd=RandomizedSearchCV(estimator=gbm,param_distributions=gbm_param_grid,scoring='accuracy',cv=3,return_train_score=True,n_jobs=-1)
                    rnd.fit(X_train,y_train)
                    y=rnd.predict(X_pred) #clasificacion
                    
            
            
                
            print("Best score is {}".format(rnd.best_score_))
            df.iloc[d_rows_na,column]=y
            
            del X_train, X_pred, y_train, y    
            self.df_clean(df_incomplete)
            self.df_clean(df_complete)
            self.df_clean(df3)
            self.df_clean(df2)
        return df
    
    #build sequential model. mlp
    def mlp_model(self,neurons_output,input_features,activ_Hidden='relu',activ_output='softmax'):
        neur_Hid_layer=self.neur_Hid_layer
        model=tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(neur_Hid_layer[0],activation=activ_Hidden,input_shape=(input_features,))) #first layer
        model.add(tf.keras.layers.BatchNormalization()) #batch normalization
        
        layers=len(neur_Hid_layer)
        
        if layers>1: #next hidden layers
            for layer in range(1,layers):
                model.add(tf.keras.layers.Dense(neur_Hid_layer[layer],activation=activ_Hidden))
                model.add(tf.keras.layers.BatchNormalization()) #batch normalization
        
        if activ_output!='linear':
            model.add(tf.keras.layers.Dense(neurons_output,activation=activ_output)) #output layer
        else:
            model.add(tf.keras.layers.Dense(neurons_output)) #output layer
        
        return model

    #compile and train model
    def train_modelmlp(self,input_features,neurons_output,X_train,y_train,activ_output='softmax',split=0.2,opt='adam',lossf='sparse_categorical_crossentropy'):
        
        epoch=self.epoch
        model=self.mlp_model(neurons_output=neurons_output,input_features=input_features,activ_output=activ_output) #obtiene el modelo
        
        if lossf=='mean_absolute_error':
            metric='mae'
            monitor_metric='val_mae'
        else:
            metric='accuracy'
            monitor_metric='val_accuracy'
        
        model.compile(optimizer=opt, loss=lossf, metrics=[metric]) #compìla el modelo
        early_stopping=EarlyStopping(monitor=monitor_metric, patience=2)
        
        model.fit(x=X_train,y=y_train, epochs=epoch, validation_data=(X_train, y_train),callbacks=[early_stopping]) #train model ,
        
        return model

    def __MLP_imputation(self, df):
        columns=self.ind_columns_na #columnas con datos faltantes
        indx_rows_complete=self.ind_rows_complete #indices de valores completos
        
        
        
        
        df2=df.copy() #crea una copia del original y si hay datos faltantes los llena con el metodo de la media o moda
        self.method='mean_mode'
        df2=self.fit_transform(df2)
        self.method='knn'
        
       
        
        flat= lambda x,num_features: x.reshape([-1, num_features]) # Flatten function
        for column in columns: #recorre las columnas con datos faltantes
            col_type=self.types_df[column] #tipo de columna
            d_rows_na=np.where(pd.isna(df.iloc[:,column]))[0]    #indices de valores faltantes en la columna de interes (y)
            
            
            df3=df2.drop([column],axis=1) #elimina la columna a evaluar (y) del df completo
            df3=pd.get_dummies(df3, drop_first=True) #elimina categoricas y crea variables dummies
            
            df_complete=df3.iloc[indx_rows_complete,:] #selecciona los ejemplos completos
            df_incomplete=df3.iloc[d_rows_na,:] #selecciona los ejemplos vacios de la columna y
            
            
            
            
            X_train=df_complete.values
            X=df_incomplete.values
            
            y_train=df.iloc[indx_rows_complete,column].values
            
            num_features=X_train.shape[1]
            num_classes = np.shape(df.iloc[indx_rows_complete,column].unique())[0]
            
            scaler = StandardScaler() #Standardize
            scaler.fit(X_train)
            X_train_S=scaler.transform(X_train)
            X_S=scaler.transform(X)
            
            X_train_flat=flat(X_train_S,num_features) #flatten
            X_flat=flat(X_S,num_features)
            
            if col_type=='float64' or col_type=='int64': #si es de tipo numérico
            
                lossf='mean_absolute_error'
                activ_output='linear'
                model=self.train_modelmlp(input_features=num_features,activ_output=activ_output,neurons_output=1,X_train=X_train_flat,y_train=y_train,lossf=lossf)
                
                y=model.predict(X_flat) #regresion
                testloss,testmetric=model.evaluate(X_train_flat,y_train) #regresa la evaluacion del modelo con el conjunto de prueba
                
            else: #categorico
                if num_classes==2:
                    yy=np.reshape(y_train,(-1, 1))
                    enc=OneHotEncoder(drop='first').fit(yy)
                    y_onehot = enc.transform(yy).toarray()
                    lossf='binary_crossentropy'
                    activ_output='sigmoid'
                    model=self.train_modelmlp(input_features=num_features,activ_output=activ_output,neurons_output=1,X_train=X_train_flat,y_train=y_onehot,lossf=lossf)
                    
                    y_pred=model.predict(X_flat) #clasificacion
                    y=enc.inverse_transform(y_pred)
                    
                    y=np.reshape(y,(y.shape[0],))
                    testloss,testmetric=model.evaluate(X_train_flat,y_onehot) #regresa la evaluacion del modelo con el conjunto de prueba
                else: #>2 
                    yy=np.reshape(y_train,(-1, 1))
                    enc=OneHotEncoder().fit(yy)
                    y_onehot = enc.transform(yy).toarray()
                    lossf='categorical_crossentropy'
                    activ_output='softmax'
                    model=self.train_modelmlp(input_features=num_features,activ_output=activ_output,neurons_output=num_classes,X_train=X_train_flat,y_train=y_onehot,lossf=lossf)
                    
                    y_pred=model.predict(X_flat) #clasificacion
                    y=enc.inverse_transform(y_pred)
                    
                    y=np.reshape(y,(y.shape[0],))
                    testloss,testmetric=model.evaluate(X_train_flat,y_onehot) #regresa la evaluacion del modelo con el conjunto de prueba
                    
                    
            print("Val score is {}".format(testmetric))
            df.iloc[d_rows_na,column]=y
            
            
            del X_train, X, y_train, y, X_train_S, X_S, X_train_flat,X_flat    
            self.df_clean(df_incomplete)
            self.df_clean(df_complete)
            self.df_clean(df3)
            self.df_clean(df2)
        return df
    
   
    

    
    def fit(self, df):
        self.__reference(df) #obtiene información del dataframe
        if self.method == 'list_wise':
            self
        elif self.method == 'LOCF':
            self
        elif self.method == 'mean_mode':
            self.__mean_mode_model(df)
        elif self.method == 'knn':
            self
        elif self.method == 'trees':
            self
        elif self.method == 'MLP':
            self 
        else:
            raise ValueError('Unrecognized method')
        
        return self
    
    def transform(self, df):
        if self.method == 'list_wise':
            df_transformed=df.dropna() #elimina renglines con datos faltantes
        elif self.method == 'LOCF':
            df_transformed=df.ffill().bfill() # llena los espacios vacios con el valor anterior
        elif self.method == 'mean_mode':
            df_transformed=self.__mean_mode_imputation(df)
        elif self.method == 'knn':
            df_transformed=self.__k_nearest_neighbors_imputation(df)
        elif self.method == 'trees':
            df_transformed=self.__trees_imputation(df) #ajusta un modelo xgboost
        elif self.method == 'MLP':
            df_transformed=self.__MLP_imputation(df) #ajusta un modelo de redes neuronales
        else:
            raise ValueError('Unrecognized method')
            
        return df_transformed
    
    def fit_transform(self, df):
        df2=df.copy()
        return self.fit(df2).transform(df2)
    
#%%
