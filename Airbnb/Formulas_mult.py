# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:18:51 2020

@author: VICTOR MANUEL GOMEZ ESPINOSA
"""

#LIBRERIAS
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
from scipy import stats
import seaborn as sns
from scipy.optimize import curve_fit, minimize
from factor_analyzer import FactorAnalyzer



#%% 
def elipse(Xmean,S,c2,label): #funcion de la elipse

    
    eigenvalues, eigenvectors = np.linalg.eigh(S) #valores y vectores propios
    eigenvectors = np.abs(eigenvectors)
    c1=np.sqrt(c2*eigenvalues) #constante
    
    xcenter, ycenter = Xmean[0],Xmean[1] #centro de la elipse
    width, height = c1[1], c1[0] #longitud de los ejes, mayor menor

    ec=np.array([1,0]) #vector canonico eje x
    angle=np.arccos(np.dot(eigenvectors[:,1],ec)) #angulo del eje mayor respecto al eje x en radianes
    angle=angle*(360/(2*np.pi)) #en grados
    if (S[0,1] <0): angle=-angle
        
   

    theta = np.deg2rad(np.arange(0.0, 360.0, 1.0)) #grados a radianes
    x = width * np.cos(theta)     #se genera la elipse centrada en el origen
    y =  height * np.sin(theta)
    v1=np.array([x, y]) #matriz de la elipse

    rtheta = np.radians(angle) #angulo a radianes
    R = np.array([                          #matriz de rotacion
        [np.cos(rtheta), -np.sin(rtheta)],
        [np.sin(rtheta),  np.cos(rtheta)],
        ])

    m=R@v1 #se  rota
    x, y = m[0],m[1]
    #x, y = np.dot(R, np.array([x, y]))
    x += xcenter #se traslada
    y += ycenter
    
    fig = plt.figure()
    plt.figure(figsize = (14,9))
    plt.plot(x,y,label=label,color="red")
    plt.plot(Xmean[0], Xmean[1],marker='o', markersize=3, color="red")
    plt.axhline(y=0, color='black', linestyle='--')
    plt.axvline(x=0, color='black', linestyle='--')
    
    plt.axis('equal')
    
    return(fig)
    
    
def distancia(X,Xmean,Sinv): #distancia
    d2=(X - Xmean)@Sinv@np.transpose(X - Xmean)
    return d2


def chi2plot(dforig,line='45'): #grafico chi2
    dfs=dforig.copy()
    Xs=dfs.values
    n,p=np.shape(Xs)
    Xmeans=np.mean(Xs,axis=0) #vector de medias
    Ss = np.cov(np.transpose(Xs)) #matriz de varianzas y covarianzas
    Ssinv=np.linalg.inv(Ss)
    ds=[]
    for i in range(0,n):
        xs=Xs[i,:]
        d=distancia(xs,Xmeans,Ssinv)
        ds.append(d)
    di=np.array(ds)
    dfs['d2']=di
    by_dist = dfs.sort_values('d2')
    p=[(i-0.5)/n for i in range(1,n+1) ]
    q=[stats.chi2.ppf(p[i], 2) for i in range(0,n) ]
    by_dist['j']=p
    by_dist['q']=q
    
    x1=np.array([5,20,35,50,75,100,200,300])
    y1=np.array([0.8788,0.9508,0.9682,0.9768,0.9838,0.9873,0.9931,0.9953])
    lf=lambda x,m,b: m*x+b
    popt, _=curve_fit(lf, x1, y1)
    m=popt[0]
    b=popt[1]
    crit=lf(n,m,b)
    corr=stats.pearsonr(by_dist.d2,by_dist.q)[0]
    if(corr<crit):
        rej='Reject H0'
    else:
        rej='No reject H0'
    #chi2 plot
    
    if line=='45':
        t = np.arange(0., max(by_dist.q), 0.2) 
        #fig = plt.figure()
        plt.figure(figsize = (14,9))
        plt.plot(t, t, 'r--') #linea a 45°
        plt.scatter(by_dist.q,by_dist.d2,label='corr: '+str(corr))
        plt.legend(loc='upper right')
        plt.xlabel('q')
        plt.ylabel('d2')
        plt.title('Chi-Square Plot')
        plt.grid()
    else:
        sns.lmplot(x='q',y='d2',data=by_dist)
        

    plt.grid()
    plt.show()
    
    
    return(by_dist,rej)
    
def boxp_bi(Xmean,S,alfa=0.05): #elipse para checar normalidad bivariada y outliers
    df=2 #2 variables
    c2=np.sqrt(stats.chi2.ppf(1-alfa, df)) #constante
    fig=elipse(Xmean,S,c2,label='Elipse de confianza del '+str((1-alfa)*100)+'%')
    return fig

#%%
def control_ch(X_Base,X_future,alpha=0.05,m=0): #tabla de control
    n,p=np.shape(X_Base)
    S=np.cov(np.transpose(X_Base))
    Sinv=np.linalg.inv(S)
    Xmean=np.mean(X_Base,0)
    
    #limit control teoric
    if(m==0): #todas las muestras
        F = stats.f.ppf(q=1.-alpha, dfn=p, dfd=n-p)
        LC = (1.0*(n-1)*p)*F/(n-p) 
        cte=((1.0*n)/(n+1))

        
        
    else:
        F = stats.f.ppf(q=1.-alpha, dfn=p, dfd=(n*m)-n-p+1)
        LC = (1.0*(n+1)*(m-1)*p)*F/((n*m)-n-p+1) 
        cte=1.0*m
    #test statistic
    Obs=np.shape(X_future)[0]
    T2 = 0
    T2vec = []
    step = []
    i = 0
    while(i<= (Obs-1)):
        d=distancia(X_future[i,:],Xmean,Sinv)
        T2 = cte*d
        T2vec.append(T2)
        step.append(i+1)
        i += 1
    stepa=np.array(step)
    T2veca=np.array(T2vec)
    Obs_T2=np.array([stepa,T2veca]).T
        
    return(Obs_T2,LC)
#%%
    
def crit_v_mean(n,p=1,t=True,alpha=0.05,mg=False): #valores criticos para el vector de medias, p=m, mg=True muestra grande(n-p>=20), t=True T, t=False F
    if(t==True):
        if(mg==False):
            alpha=alpha/(2*p)
            T=stats.t.ppf(1-alpha, n-1)
        else: #aproximacion para muestras grandes
            T=stats.norm.ppf(1-alpha,  loc=0, scale=1)
        cv=T
    
    else:    
        if(mg==False):
            F = stats.f.ppf(q=1.-alpha, dfn=p, dfd=n-p)
            F = (1.0*(n-1)*p)*F/(n-p)
        else: #aproximacion para muestras grandes
            F=stats.chi2.ppf(1-alpha, p)
        cv=F
    
    return(cv)

def verosim_vmean_test(mu,X,alpha=0.05,mg=False): #test de razon de versosimilitud para vector de medias H0
    n,p = np.shape(X)
    X_bar=np.mean(X,axis=0)
    S = np.cov(np.transpose(X))
    Sinv=np.linalg.inv(S)
    
    d=distancia(X_bar,mu,Sinv)
    cte=n
    #test statistic
    T2t=cte*d
    
    Ae=(1+(T2t/(n-1)))**(-1) #lambda de Wilks (estadistico de prueba)
    
    #valores criticos
    t=False
    T2c=crit_v_mean(n,p,t,alpha,mg)
    Ac=(1+(T2c/(n-1)))**(-1) #valor critico de la razon de verosimilitud
    
    val=np.array([Ae,Ac]).T
    if Ae>Ac:
        rej='No rechaza H0'
    else:
        rej='Rechaza H0'
    
    return(val,rej)
    
def confi_elip_vmean(Xmean,S,n,alpha=0.05,mg=False): #elipse de confianza simultanea para la media
    p=2 #2 variables
    t=False
    c2=crit_v_mean(n,p,t,alpha,mg) #constante
    fig=elipse(Xmean,S,c2,label='Elipse de confianza simultanea para la media, del '+str((1-alpha)*100)+'%')
    return fig

def confi_int_vmean(Xmean,S,n,p,Bonfe=True,alpha=0.05,mg=False): #intervalos de confianza para la media
    cte=(1/n)
    diag=cte*np.diag(S)
    
    if(Bonfe==True):
        t=True
        vc=crit_v_mean(n,p,t,alpha,mg) #constante
        raiz=np.sqrt(diag)
        l=Xmean-vc*raiz
        u=Xmean+vc*raiz
    else:
        t=False
        vc=crit_v_mean(n,p,t,alpha,mg) #constante
        raiz=np.sqrt(vc*diag)
        l=Xmean-raiz
        u=Xmean+raiz
        
    interv=np.array([l,u]).T
    
    return(interv)
        
#%%    
def H0_mu1_m2_test(X1,X2,alpha=0.05,mg=False, same_sig12=True): #test para diferencia de medias H0
    n1,p1 = np.shape(X1)
    n2,p2 = np.shape(X1)
    p=p1
    
    X_bar1=np.mean(X1,axis=0)
    X_bar2=np.mean(X2,axis=0)
    X_bardif=X_bar1-X_bar2
    mudif=X_bardif*0
    
    S1 = np.cov(np.transpose(X1))
    S2 = np.cov(np.transpose(X2))
    spooled=((n1-1)*S1+(n2-1)*S2)/(n1+n2-2)
    cte=((1/n1)+(1/n2))
    spooled=cte*spooled
    
    if(same_sig12==False):
        spooled=(S1/n1)+(S2/n2)
    
    Sinv=np.linalg.inv(spooled)
    
    
    
    
    
    d=distancia(X_bardif,mudif,Sinv)
    
    #test statistic
    T2t=d
    
    #valores criticos
    n=n1+n2-1
    t=False
    T2c=crit_v_mean(n,p,t,alpha,mg)
    
    
    val=np.array([T2t,T2c]).T
    if T2t>T2c:
        rej='Rechaza H0'
    else:
        rej='No rechaza H0'
    
    ap=(1/100)*Sinv@X_bardif.T #mas responsables
    
    return(val,rej,ap)

def confi_elip_difmean(Xdif,spooled,n1,n2,alpha=0.05,mg=False): #elipse de confianza simultanea para la diferencia de medias
    p=2 #2 variables
    n=n1+n2-1
    cte=((1/n1)+(1/n2))
    t=False
    c2=cte*crit_v_mean(n,p,t,alpha,mg)
    fig=elipse(Xdif,spooled,c2,label='Elipse de confianza para mu1-mu2, del '+str((1-alpha)*100)+'%')
    return fig

def confi_int_difmean(X1,X2,p,Bonfe=True,alpha=0.05,mg=False): #intervalos de confianza para la dif medias
    n1,p1 = np.shape(X1)
    n2,p2 = np.shape(X1)
    #p=p1
    
    X_bar1=np.mean(X1,axis=0)
    X_bar2=np.mean(X2,axis=0)
    X_bardif=X_bar1-X_bar2
    
    S1 = np.cov(np.transpose(X1))
    S2 = np.cov(np.transpose(X2)) 
    spooled=((n1-1)*S1+(n2-1)*S2)/(n1+n2-2)
    
    cte=((1/n1)+(1/n2))
    diag=cte*np.diag(spooled)
    n=n1+n2-1
    
    if(Bonfe==True):
        t=True
        vc=crit_v_mean(n,p,t,alpha,mg) #constante
        raiz=np.sqrt(diag)
        l=X_bardif-vc*raiz
        u=X_bardif+vc*raiz
    else:
        t=False
        vc=crit_v_mean(n,p,t,alpha,mg) #constante
        raiz=np.sqrt(vc*diag)
        l=X_bardif-raiz
        u=X_bardif+raiz
        
    interv=np.array([l,u]).T
    
    return(interv)
#%%    
def statisticsf(W_E,B_H): #multivariate test statistics 
    Wilks=np.linalg.det(W_E)/np.linalg.det(B_H+W_E) #B+W Total
    Lawley_H=np.trace(B_H@np.linalg.inv(W_E))
    Pillai_t=np.trace(B_H@np.linalg.inv(B_H+W_E))
    
    m=W_E@np.linalg.inv(B_H+W_E)
    eigval,eigvec = np.linalg.eigh(m)
    
    Roys_lr=eigval[-1]
    return(Wilks,Lawley_H,Pillai_t,Roys_lr)
    
def diststat(W_E,B_H,p,pob,sampleT, op=1, alfa=0.05, mg=False): #test MANOVA op=1 Wilks, op=2 Lawley_H, op=3 Pillai_t, op=4 Roys_lr
    
    Wilks,Lawley_H,Pillai_t,Roys_lr=statisticsf(W_E,B_H)
    
    nt=sampleT
    g=pob
    h=g-1
    e=nt-g
    if op==1: #Wilks
        lamb=Wilks
        if(mg==False):
            if(p==1 and g>=2):
                df1=g-1
                df2=nt-g
                f=stats.f.ppf(q=1-alfa, dfn=df1, dfd=df2)
    
                stt=((nt-g)/(g-1))*((1-lamb)/lamb)
            elif(p==2 and g>=2):
                df1=2*(g-1)
                df2=2*(nt-g-1)
                f=stats.f.ppf(q=1-alfa, dfn=df1, dfd=df2)
    
                stt=((nt-g-1)/(g-1))*((1-np.sqrt(lamb))/np.sqrt(lamb))
            elif(p>=1 and g==2):
                df1=p
                df2=nt-p-1
                f=stats.f.ppf(q=1-alfa, dfn=df1, dfd=df2)
    
                stt=((nt-p-1)/(p))*((1-lamb)/lamb)
            elif(p>=1 and g==3):
                df1=2*p
                df2=2*(nt-p-2)
                f=stats.f.ppf(q=1-alfa, dfn=df1, dfd=df2)
    
                stt=((nt-p-2)/(p))*((1-np.sqrt(lamb))/np.sqrt(lamb))
            else:
                stt=-1*(nt-1-((p+g)/2))*np.log(lamb)
    
                df=p*(g-1)
                f=stats.chi2.ppf(1-alfa, df)
        else:
            stt=-1*(nt-1-((p+g)/2))*np.log(lamb)
            
            df=p*(g-1)
            f=stats.chi2.ppf(1-alfa, df)
            
        vH0=stt>f
        
    elif op==2: #Lawley-Hotelling
        lamb=Lawley_H
        a=p*h
        B=((e+h-p-1)*(e-1))/((e-p-3)*(e-p))
        b=4+(a+2)/(B-1)
        c=a*(b-2)/(b*(e-p-1))
        
        df1=a
        df2=b
        f=stats.f.ppf(q=1-alfa, dfn=df1, dfd=df2)
        
        
        stt=lamb/(c*e)
        vH0=stt<f
        
    elif op==3: #Pillai
        lamb=Pillai_t
        s=min(p,h)
        m=(abs(p-h)-1)/2
        n=(e-p-1)/2
        
        df1=s*(2*m+s+1)
        df2=s*(2*n+s+1)
        f=stats.f.ppf(q=1-alfa, dfn=df1, dfd=df2)
        
        stt=(lamb*(2*n+s+1))/((2*m+s+1)*(s-lamb))
        vH0=stt>f
        
    elif op==4: # Roy's
        lamb=Roys_lr
        s=min(p,h)
        v1=(abs(p-h)-1)/2
        v2=(e-p-1)/2
        
        df1=2*v1+2
        df2=2*v2+2
        f=stats.f.ppf(q=1-alfa, dfn=df1, dfd=df2)
        
        stt=(2*v2+2)*lamb/(2*v1+2)
        vH0=stt>f

    
    #vH0=stt>f #vh0 True, reject H0
    
    if(vH0==True):
        res='Reject H0'
    else:
        res='Accept H0'
    
    return (stt,f,res)
#%%
def get_X_S_bi(Xbar,S,var1,var2): #toma la media y matriz covarianzas para 2 variables (var1,var2) de una matriz p -variables
    v1=var1
    v2=var2
    v1=v1-1 #variables
    v2=v2-1
    x_mean=np.array([Xbar[v1],Xbar[v2]]) 
    S1=np.array([[S[v1,v1],S[v1,v2]],[S[v1,v2],S[v2,v2]]])
    
    return(x_mean,S1)
    
    
def confi_elip_difmeandifsig(Xdif,s1,s2,n1,n2,alpha=0.05,mg=False): #elipse de confianza simultanea para la diferencia de medias cuando las cov son diferentes y muestras grandes
    p=2 #2 variables
    s3=(s1/n1)+(s2/n2)
    c2=np.sqrt(stats.chi2.ppf(1-alpha, p)) #constante
    fig=elipse(Xdif,s3,c2,label='Elipse de confianza para mu1-mu2, del '+str((1-alpha)*100)+'%')
    return fig


def confi_int_difmeandifsig(X1,X2,p,Bonfe=True,alpha=0.05,mg=False): #intervalos de confianza para la dif medias para muestras grandes cuando la cov son diferentes
    n1,p1 = np.shape(X1)
    n2,p2 = np.shape(X1)
    p=p1
    
    X_bar1=np.mean(X1,axis=0)
    X_bar2=np.mean(X2,axis=0)
    X_bardif=X_bar1-X_bar2
    
    S1 = np.cov(np.transpose(X1))
    S2 = np.cov(np.transpose(X2)) 
    s3=(S1/n1)+(S2/n2)
    
    
    diag=np.diag(s3)
  
    vc=np.sqrt(stats.chi2.ppf(1-alpha, p)) #constante
    raiz=np.sqrt(vc*diag)
    l=X_bardif-raiz
    u=X_bardif+raiz
        
    interv=np.array([l,u]).T
    
    return(interv)
#%%  
def linregc(y,z): #regresion lineal clasica
    h=np.linalg.inv(z.T@z)@z.T
    H=z@h
    n2,p2=np.shape(H)
    I=np.identity(n2)
    
    betas=h@y #B
    resid=(I-H)@y #e
    sumsqres=resid.T@resid
    e2=sumsqres
    
    yb=np.mean(y,0)
    yb2=yb**2
    ys2=(y.T@y)-(len(y)*yb2)
    R2=1-(e2/ys2)
    return(betas,resid,sumsqres,R2)
    
def confi_elip_betas(y,z,alpha=0.05,mg=False): #elipse de confianza simultanea para las betas
    n,p=np.shape(z)
    r=p-1
    
    h=np.linalg.inv(z.T@z)@z.T
    H=z@h
    n2,p2=np.shape(H)
    I=np.identity(n2)
    
    betas=h@y
    resid=(I-H)@y
    sumsqres=resid.T@resid
    
    ssq=(sumsqres)/(n-(r+1))
    s3=ssq*(np.linalg.inv(z.T@z))
    t=True
    n=n-r
    c2=crit_v_mean(n,1,t,alpha,mg) #constante
    fig=elipse(betas,s3,c2,label='Elipse de confianza para betas, del '+str((1-alpha)*100)+'%')
    return fig


def confi_int_betas(y,z,alpha=0.05,mg=False): #intervalos de confianza para las betas
    n,p=np.shape(z)
    r=p-1
    
    h=np.linalg.inv(z.T@z)@z.T
    H=z@h
    n2,p2=np.shape(H)
    I=np.identity(n2)
    
    betas=h@y
    resid=(I-H)@y
    sumsqres=resid.T@resid
    
    ssq=(sumsqres)/(n-(r+1))
    s3=ssq*(np.linalg.inv(z.T@z))
    
    
    diag=np.diag(s3)
    t=True
    n=n-r
    vc=crit_v_mean(n,1,t,alpha,mg)#constante
    raiz=np.sqrt(diag)
    l=betas-vc*raiz
    u=betas+vc*raiz
        
    interv=np.array([l,u]).T
    
    return(interv)


    
def ver_betas_test(y,z,ztest,alpha=0.05): #test para probar Beta_ztest=0 
    v=ztest
    q=v-1
    n,p=np.shape(z)
    r=p-1
    c=np.arange(0,p)
    colummns=c!=v
    z1=z[:,colummns]
    
    betas,resid,sumsqres,R2=linregc(y,z)
    betas1,resid1,sumsqres1,R21=linregc(y,z1)
    
    ssq=(sumsqres)/(n-(r+1))
    
    est=((sumsqres1-sumsqres)/(r-q))/ssq
    
    df1=r-q
    df2=n-r-1
    f=stats.f.ppf(q=1-alpha, dfn=df1, dfd=df2)
    
    if(est>f):
        rej='Reject H0'
    else:
        rej='No Reject H0'
        
    val=np.array([est,f])
    return(val,rej)
    
#%%
def linregMult(y,z): #regresion lineal multivariada
    h=np.linalg.inv(z.T@z)@z.T
    H=z@h
    n2,p2=np.shape(H)
    I=np.identity(n2)
    
    betas=h@y #B
    resid=(I-H)@y #e
    sumsqres=resid.T@resid
    
    
    
    return(betas,resid,sumsqres)

def Manova_betas_test(y,z,ztest,alpha=0.05): #test para probar H0: B_ztest=0
    v=ztest
    q=v-1
    
    n,p=np.shape(z)
    r=p-1
    c=np.arange(0,p)
    colummns=c!=v
    z1=z[:,colummns]
    
    p,m=np.shape(y)
    
    
    
    betas,resid,E=linregMult(y,z)
    betas1,resid1,E1=linregMult(y,z1)
    
    H=E1-E
    lamb=np.linalg.det(E)/np.linalg.det(E+H)
    
    stt=-1*(n-r-1-0.5*(m-r+q+1))*np.log(lamb)
    dof=m*(r-q)
    f=stats.chi2.ppf(1-alpha, dof)
    
    if(stt>f):
        rej='Reject H0'
    else:
        rej='No reject H0'
    
    
    val=np.array([lamb,stt,f])
    
    return(val,rej)

def confi_elip_predM(y,z,z0,alpha=0.05): #elipse de confianza simultanea para prediccion
    
    
    
    n,p=np.shape(z)
    r=p-1
    
    p,m=np.shape(y)
    

    betas,resid,E=linregMult(y,z)
    
    df1=m
    df2=n-r-m
    f=stats.f.ppf(q=1-alpha, dfn=df1, dfd=df2)
    F=(m*(n-r-1))*f/(n-r-m)
    cte=1+(z0.T@np.linalg.inv(z.T@z)@z0)
    c2=cte*F
    
    s=E/(n-r-1)
    
    v=z0.T@betas
    
    
    
    
    
    fig=elipse(v,s,c2,label='Elipse de predicción, del '+str((1-alpha)*100)+'%')
    return fig

def confi_elip_bz0(y,z,z0, alpha=0.05,mg=False): #elipse de confianza simultanea para las betas
    n,p=np.shape(z)
    r=p-1
    
    p,m=np.shape(y)
    

    betas,resid,E=linregMult(y,z)
    
    df1=m
    df2=n-r-m
    f=stats.f.ppf(q=1-alpha, dfn=df1, dfd=df2)
    F=(m*(n-r-1))*f/(n-r-m)
    cte=(z0.T@np.linalg.inv(z.T@z)@z0)
    c2=cte*F
    
    s=E/(n-r-1)
    
    v=z0.T@betas
    
    
    
    
    
    fig=elipse(v,s,c2,label='Elipse de confianza para bz0, del '+str((1-alpha)*100)+'%')
    
    return fig

def confi_int_predM(y,z,z0,alpha=0.05): #intervalo de confianza simultanea para prediccion
    
    
    
    n,p=np.shape(z)
    r=p-1
    
    p,m=np.shape(y)
    

    betas,resid,E=linregMult(y,z)
    
    df1=m
    df2=n-r-m
    f=stats.f.ppf(q=1-alpha, dfn=df1, dfd=df2)
    F=(m*(n-r-1))*f/(n-r-m)
    cte=1+(z0.T@np.linalg.inv(z.T@z)@z0)
    c2=cte*F
    
    s=E/(n-r-1)
    
    v=z0.T@betas
    
    
    
    diag=np.diag(s)
    
    raiz=np.sqrt(c2*diag)
    l=v-raiz
    u=v+raiz
        
    interv=np.array([l,u]).T
    
    
    return interv

#%% boxcox multivariado
def fun(lambda2,x,m): #funcion necesaria para  boxcoxlamb
    vl=np.log(x)
    la=(x**np.tile(lambda2,(m,1))-1)/np.tile(lambda2,(m,1))
    S=np.cov(la.T)
    f=((m/2)*(np.log(np.linalg.det(S))))-((lambda2-1)@(np.sum(vl,0)))
    return(f)

def boxcoxlamb(x): # lambas para la transformacion boxcox vibariada
    m,n=x.shape
    lambda_ini=np.zeros(n)
    pt1 = PowerTransformer(method='box-cox') 
    pt1.fit(X)
    lambdas1=pt1.lambdas_
    lambda_ini=lambdas1
    #fun = lambda (lambda2,x,m): (((m/2)*(np.log(np.linalg.det((np.cov(((x**np.matlib.repmat(lambda2.T,m,1)-1)/np.matlib.repmat(lambda2.T,m,1)))))))-((lambda2-1).T*(np.sum(np.log(x))).T))
    
    lambda2=minimize(fun,lambda_ini,args=(x,m))
    return(lambda2.x)

#transformacion de boxcox univariada y multivariada (solo datos positivos)
def boxcoxtransf(x,lamb): 
    if lamb==0:
        trans=np.log(x)
    else:
        trans=((x**lamb)-1)/lamb
    return(trans)

def boxcoxtr(x,lamb):
    n,p=x.shape
    transformdata=np.zeros((n,p))
    if(np.sum(x<0)==0): #boxcox solo para datos positivos
        for i in range(p):
            transformdata[:,i]=boxcoxtransf(x[:,i],lamb[i])
    return(transformdata)



#%% factores
def numbFactorsTest(X,m=1,met='ml',alfa=0.05): #met='principal','minres'
    n,p=X.shape
    R = np.corrcoef(np.transpose(X))
    p_val=0
    
    fa=FactorAnalyzer(method=met,rotation='varimax',n_factors=m,is_corr_matrix=False)
    fa.fit(X)
    l=fa.loadings_
    ll=l@l.T
    fi=np.diag(R)- np.diag(ll)
    Sg=ll+np.diag(fi)
    
    l=1/2*(2*p+1-(8*p+1)**0.5)
    if m<l:
        df=(((p-m)**2) - (p+m))*1/2
        vt=(n-1-(2*p+4*m+5)/6)*np.log(np.linalg.det(Sg)/np.linalg.det(R))
        vc=stats.chi2.ppf(1-alfa, df)
        p_val=stats.chi2.pdf(vt,df,1-alfa) #p-value
        if vt>vc: #se rechaza H0
            H0=False
        else:
            H0=True
    else:
        H0=False
    
    cumVar=fa.get_factor_variance()[2][-1]
    return(H0,p_val,cumVar)

#%% CA y MCA

def escala(x): #puntuaciones
    y=(x-min(x))*10/(max(x)-min(x))
    return(y)

def independenciaCA(df,alpha=0.05): #independencia entre filas y columnas (True)
    I,J=df.shape
    fisher_tab=df.values
    n=np.sum(fisher_tab)
    P=fisher_tab/n
    r = np.sum(P,1)
    c = np.sum(P,0)
    
    vc=stats.chi2.ppf(1-alpha, (I-1)*(J-1))
    ren=0
    for i in range(I):
        col=0
        for j in range(J):
            col=col+(((P[i,j]/r[i])-c[j])**2)*(1/c[j])
        ren=ren+r[i]*col

    vt=n*ren
    return(vt<=vc,(vt,vc))