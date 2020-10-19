# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 19:17:50 2020

EQUIPO GOODBATCH
VICTOR MANUEL GOMEZ ESPINOSA
"""
#librerias
import tensorflow as tf
import numpy as np
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator
from scipy.stats import mode

#Clase que regresa el sentimiento de un comentario en Tweeter
class comentClassifier(BaseEstimator):
    def __init__(self,model = 'hyb',bagging=['svm','xgb','mlp']):
        self.model = model
        self.bagging = bagging
                

    def Sentimiento(self,texto):
        AnSent = SentimentIntensityAnalyzer()
        sentiment_dict = AnSent.polarity_scores(texto)  
        # decide sentiment as positive, negative and neutral 
        if sentiment_dict['compound'] > 0.0 : 
            decide = "P"
      
        elif sentiment_dict['compound'] <  0.0 : 
            decide = "N"
      
        else : 
            decide = "NEU"
    
        return decide

    def predSent(self,textVector):
        labels=[]
        for text in textVector:
            label=self.Sentimiento(text)
            labels.append(label)
        return np.array(labels)
    
    def __vader_pred(self, X):
        y_vader=self.predSent(X) #vader
        
        return y_vader

    def __svm_pred(self, X):
        
        text_model = joblib.load('text_model.sav')
        X=text_model.transform(X) #matriz de terminos
        
        SVM_model = joblib.load('SVM_model.sav')

        y_svm=SVM_model.predict(X)

        return y_svm
    
    def __xgb_pred(self, X):
        
        text_model = joblib.load('text_model.sav')
        X=text_model.transform(X) #matriz de terminos
        
        XGB_model = joblib.load('XGB_model.sav')

        y_xgb=XGB_model.predict(X)

        return y_xgb
    
    def __mlp_pred(self, X):
        text_model = joblib.load('text_model.sav')
        X=text_model.transform(X) #matriz de terminos
        
        MLP_model = tf.keras.models.load_model('MLP1.h5') #carga el modelo
        onehot_model = joblib.load('onehot_model.sav')
        
        y_mlp=MLP_model.predict(X)
        y_mlp=onehot_model.inverse_transform(y_mlp)
        
        return y_mlp[:,0]
    
    def __bagging_pred(self, X):
        n=len(self.bagging)
        predictions = np.empty((X.shape[0], n), dtype='object')
        
        i=0
        for model in self.bagging:
            self.model=model
            predictions[:, i] = self.predict(X)
            i+=1
            
        MajorityClass = mode(predictions, axis = 1)[0].ravel()
        y_vader=self.__vader_pred(X)
        predictions = np.empty((X.shape[0], 2), dtype='object')
        predictions[:, 0] = y_vader
        predictions[:, 1] = MajorityClass
        MajorityClass = mode(predictions, axis = 1)[0].ravel()
        
        
        #MajorityClass[MajorityClass=='NEU']=MajorityClass2[MajorityClass=='NEU']
        
        
        return MajorityClass
    
    
    def predict(self, X):
        #classifier
        if self.model == 'vader':
            return self.__vader_pred(X)
        elif self.model == 'svm':
            return self.__svm_pred(X)
        elif self.model == 'xgb':
            return self.__xgb_pred(X)
        elif self.model == 'mlp':
            return self.__mlp_pred(X)
        elif self.model == 'hyb':
            return self.__bagging_pred(X)
        else:
            raise ValueError('Undefined method')
            
        return self
        
            

