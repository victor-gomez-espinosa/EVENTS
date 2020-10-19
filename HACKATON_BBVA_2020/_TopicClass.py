# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 14:51:12 2020

EQUIPO GOODBATCH
VICTOR MANUEL GOMEZ ESPINOSA
"""
#librerias
import numpy as np
from microtc.textmodel import TextModel
from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

#Clase que clasifica comentarios en categorías, segun palabras clave
class TopicClassifier(BaseEstimator):
    def __init__(self,categories = ['productos','digital','atn_cliente','seguridad','sucursal'],topics=['tarjetas de credito, debito, credito de nomina, personal, automotriz, hipotecario, seguros, inversiones','app, aplicacion movil, banca en linea, biometricos, datos, token movil, reconocimiento facial, voz', 'atencion, servicio al cliente', 'seguridad, estafa, fraude, robo', 'cajero atumatico, tiempo de espera, filas, sucio, sucursal, cajas, ejecutivos, muchos papeles' ]):
        self.categories = categories
        self.topics = topics
         
    
    def maxin(self,xx): #selecciona la categoria con el mayor score
        maxi=xx==max(xx)
        maxs=np.sum(maxi)
        if maxs==1: a=xx.argmax()
        else: a=np.random.choice(np.where(xx==max(xx))[0])
        return(int(a))

    
    def fit_predict(self, X):
        #fit
        #Corpus
        textmodel = TextModel().fit(X) #Modelo de texto
        X=textmodel.transform(X) 
        
        nmf=NMF(n_components=6, max_iter=500).fit(X) #reduccion de dimension
        nmf_features=nmf.transform(X)
        
        #Topics
        topics=self.topics
        X_topics=textmodel.transform(topics) #al modelo de texto
        nmf_topics=nmf.transform(X_topics) #reduccion de dimension
        
        K=cosine_similarity(nmf_features,nmf_topics) #similaridades
        n,p=K.shape
        
        #predict
        prob=K
        cat=self.categories
        labels=[]
        for i in range(prob.shape[0]):
            xx=prob[i,:]
            indx=self.maxin(xx)
            lab=cat[indx]
            labels.append(lab)
        
        
        return np.array(labels)
    