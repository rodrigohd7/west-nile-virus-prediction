# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 15:42:45 2018

@author: aline
"""
from yearly_data import YearlyData
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import scipy.stats as sc

#AMOSTRA NÃO BALANCEADA
#def Logistic2(x_train,y_train,x_test): 
#    classifier = LogisticRegression(random_state=0)  
#    classifier.fit(x_train,y_train)
#    y_pred_train = classifier.predict_proba(x_train)[:,1]
#    y_pred_class = classifier.predict(x_train)
#    y_pred_test = classifier.predict_proba(x_test)[:,1]   
#    confusion_matrix=pd.crosstab(y_train, y_pred_class, rownames=['True'], colnames=['Predicted'], margins=True)
#    print("Treino \n", confusion_matrix)
#    print("Treino \n", classifier.score(df_train,y_train))
#    return y_pred_test

def Logistic(x_train,y_train, x_test): 
    classifier = LogisticRegression()  
    classifier.fit(x_train,y_train)
    #y_pred_train = classifier.predict(x_train)
    y_pred_train = classifier.predict_proba(x_train)[:,1]  
    y_pred_trainf =getNormalizedTarget(y_pred_train)
    #y_pred_test = classifier.predict(x_test)
    y_pred_test_kaggle = classifier.predict_proba(x_test)[:,1]  
    #=confusion_matrix( np.array(y_train), np.array(y_pred_train))
    confusion_matrix=pd.crosstab(y_train, y_pred_trainf, rownames=['True'], colnames=['Predicted'], margins=True)
    print("Treino \n", confusion_matrix)
    print("Treino \n", accuracy_score(y_train,y_pred_trainf))
    print("Treino \n", classification_report(y_train,y_pred_trainf))
    return y_pred_test_kaggle

def getNormalizedTarget(array_target):
	for index in range(len(array_target)):
		array_target[index] = 1 if array_target[index] >= np.mean(array_target) else 0
		# array_target[index] = int(array_target[index])
	return array_target

#escalonando normalização
def norm(df_train):
    df_train2=df_train
    std_scale = preprocessing.StandardScaler().fit(df_train2[['Tavg','Tmin','Tmax','Year','WetBulb', 'Sunset', 'Sunrise', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'PrecipTotal', 'Month', 'Longitude', 'Latitude', 'Heat', 'DewPoint', 'Day', 'Cool', 'Block', 'AvgSpeed','AddressAccuracy']])
    df_std = std_scale.transform(df_train2[['Tavg','Tmin','Tmax','Year','WetBulb', 'Sunset', 'Sunrise', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'PrecipTotal', 'Month', 'Longitude', 'Latitude', 'Heat', 'DewPoint', 'Day', 'Cool', 'Block', 'AvgSpeed','AddressAccuracy']])
    df_std = pd.DataFrame(df_std)
    df_std.columns=['Tavg','Tmin','Tmax','Year','WetBulb', 'Sunset', 'Sunrise', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'PrecipTotal', 'Month', 'Longitude', 'Latitude', 'Heat', 'DewPoint', 'Day', 'Cool', 'Block', 'AvgSpeed','AddressAccuracy']
    cols=list(df_std)
    df_cat=df_train2.drop(df_train2[cols],axis=1)
    df_cat = df_cat.reset_index(drop=True)
    #merged_left = pd.merge(left=surveySub,right=speciesSub, how='left', left_on='species', right_on='species_id')
    frames1=[df_std,df_cat]
    df_norm=pd.concat(frames1,axis=1)
    return df_norm

#min_max
def min_max(df_train):
    df_train2=df_train
    minmax_scale = preprocessing.MinMaxScaler().fit(df_train2[['Tavg','Tmin','Tmax','Year','WetBulb', 'Sunset', 'Sunrise', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'PrecipTotal', 'Month', 'Longitude', 'Latitude', 'Heat', 'DewPoint', 'Day', 'Cool', 'Block', 'AvgSpeed','AddressAccuracy']])
    df_minmax = minmax_scale.transform(df_train2[['Tavg','Tmin','Tmax','Year','WetBulb', 'Sunset', 'Sunrise', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'PrecipTotal', 'Month', 'Longitude', 'Latitude', 'Heat', 'DewPoint', 'Day', 'Cool', 'Block', 'AvgSpeed','AddressAccuracy']])
    df_minmax = pd.DataFrame(df_minmax)
    df_minmax.columns=['Tavg','Tmin','Tmax','Year','WetBulb', 'Sunset', 'Sunrise', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'PrecipTotal', 'Month', 'Longitude', 'Latitude', 'Heat', 'DewPoint', 'Day', 'Cool', 'Block', 'AvgSpeed','AddressAccuracy']
    cols=list(df_minmax)
    df_cat=df_train2.drop(df_train2[cols],axis=1)
    df_cat = df_cat.reset_index(drop=True)
    frames2=[df_minmax,df_cat]
    df_mm=pd.concat(frames2,axis=1)
    return df_mm
   
#SELECIONANDO VARIAVEIS
def feat_imp(df_train,y_train, n_feat):
    model = ExtraTreesClassifier()
    model.fit(df_train, y_train)
    indices = np.argsort(model.feature_importances_)[::-1]
    indices = indices[1:n_feat]
    n_df=df_train[df_train.columns[indices]]
    return n_df

def sel_rfe(x,y,n):
    model = LogisticRegression()
    # create the RFE model and select 3 attributes
    rfe = RFE(model, n)
    rfe = rfe.fit(x, y)
    # summarize the selection of the attributes
    #print(rfe.support_)
    #print(rfe.ranking_)
    indices= [i for i, z in enumerate(rfe.support_) if z]
    n_df=x[x.columns[indices]]
    return n_df    

def export(ids_test,y_pred_test1,name):
    df_result = pd.DataFrame()
    df_result.insert(loc=0, column='Id', value=ids_test)
    df_result.insert(loc=1, column='WnvPresent', value=y_pred_test1)
    df_result.to_csv(name+".csv", index=False)
    

yearly = YearlyData("C://Users//aline//Downloads//MO444//Final Work//west_nile//input//train.csv","C://Users//aline//Downloads//MO444//Final Work//west_nile//input//weather.csv")
data1 = yearly.process_for(2007)
data2 = yearly.process_for(2009)
data3 = yearly.process_for(2011)
data4 = yearly.process_for(2013)
yearly2 = YearlyData("C://Users//aline//Downloads//MO444//Final Work//west_nile//input//test.csv","C://Users//aline//Downloads//MO444//Final Work//west_nile//input//weather.csv")
data5 = yearly2.process_for(2008)
data6 = yearly2.process_for(2010)
data7 = yearly2.process_for(2012)
data8 = yearly2.process_for(2014)

#TREINO
y_data1=data1["WnvPresent"]
y_data2=data2["WnvPresent"]
y_data3=data3["WnvPresent"]
y_data4=data4["WnvPresent"]
#data1=data1.drop(["WnvPresent", "NumMosquitos","Year"],axis=1)
#data2=data2.drop(["WnvPresent", "NumMosquitos","Year"],axis=1)
#data3=data3.drop(["WnvPresent", "NumMosquitos","Year"],axis=1)
#data4=data4.drop(["WnvPresent", "NumMosquitos","Year"],axis=1)
data1=data1.drop(["WnvPresent", "NumMosquitos"],axis=1)
data2=data2.drop(["WnvPresent", "NumMosquitos"],axis=1)
data3=data3.drop(["WnvPresent", "NumMosquitos"],axis=1)
data4=data4.drop(["WnvPresent", "NumMosquitos"],axis=1)
frames_train = [data1,data2,data3,data4]
df_train=pd.concat(frames_train)
frames_ytrain =[y_data1,y_data2,y_data3,y_data4]
y_train=pd.concat(frames_ytrain)
#as variaveis que não existem de um ano para outro são preenchidas com 0 para que não impacte o processo de modelagem
#foram preenchidas com este valor porque são variáveis categóricas com 2 levels = 0 ou 1
df_train=df_train.fillna(0)

#TESTE
frames_test = [data5,data6,data7,data8]
df_test=pd.concat(frames_test)
#as variaveis que não existem de um ano para outro são preenchidas com 0 para que não impacte o processo de modelagem
#foram preenchidas com este valor porque são variáveis categóricas com 2 levels = 0 ou 1
df_test=df_test.fillna(0)
ids_test = np.array(df_test['Id'])
df_test=df_test.drop('Id',axis=1)

#LOGISTICA TODAS AS VARIAVEIS
#mantendo mesmas colunas pro teste
y_pred_test1=Logistic(df_train,y_train,df_test[list(df_train.columns)])
export(ids_test,y_pred_test1,"log_tot")

#logistica normalizada
df_norm=norm(df_train)
y_pred_norm=Logistic(df_norm,y_train,df_test[list(df_train.columns)])
export(ids_test,y_pred_norm,"log_norm")

#logistica min max
df_mm=min_max(df_train)
y_pred_mm=Logistic(df_mm,y_train,df_test[list(df_train.columns)])
export(ids_test,y_pred_mm,"log_mm")

#selecionando variaveis
#RFE
df_sel_train=sel_rfe(df_train, y_train,30)
y_pred_sel_train=Logistic(df_sel_train,y_train,df_test[list(df_sel_train.columns)])
export(ids_test,y_pred_sel_train,"sel_train")
df_sel_train2=sel_rfe(df_train, y_train,20)
y_pred_sel_train2=Logistic(df_sel_train2,y_train,df_test[list(df_sel_train2.columns)])
export(ids_test,y_pred_sel_train2,"sel_train2")
df_sel_train3=sel_rfe(df_train, y_train,10)
y_pred_sel_train3=Logistic(df_sel_train3,y_train,df_test[list(df_sel_train3.columns)])
export(ids_test,y_pred_sel_train3,"sel_train3")

#trees
df_sel_train4=feat_imp(df_train, y_train,30)
y_pred_sel_train4=Logistic(df_sel_train4,y_train,df_test[list(df_sel_train4.columns)])
export(ids_test,y_pred_sel_train4,"sel_train4")
df_sel_train5=feat_imp(df_train, y_train,20)
y_pred_sel_train5=Logistic(df_sel_train5,y_train,df_test[list(df_sel_train5.columns)])
export(ids_test,y_pred_sel_train5,"sel_train5")
df_sel_train6=feat_imp(df_train, y_train,10)
y_pred_sel_train6=Logistic(df_sel_train6,y_train,df_test[list(df_sel_train6.columns)])
export(ids_test,y_pred_sel_train6,"sel_train6")
df_sel_train17=feat_imp(df_train, y_train,200)
y_pred_sel_train6=Logistic(df_sel_train17,y_train,df_test[list(df_sel_train17.columns)])
export(ids_test,y_pred_sel_train6,"sel_train7")
df_sel_train8=feat_imp(df_train, y_train,150)
y_pred_sel_train6=Logistic(df_sel_train8,y_train,df_test[list(df_sel_train8.columns)])
export(ids_test,y_pred_sel_train6,"sel_train8")
df_sel_train9=feat_imp(df_train, y_train,100)
y_pred_sel_train6=Logistic(df_sel_train9,y_train,df_test[list(df_sel_train9.columns)])
export(ids_test,y_pred_sel_train6,"sel_train9")
df_sel_train10=feat_imp(df_train, y_train,50)
y_pred_sel_train6=Logistic(df_sel_train10,y_train,df_test[list(df_sel_train10.columns)])
export(ids_test,y_pred_sel_train6,"sel_train10")

#ANALISE ANOS
#todas as variáveis
y_pred_d1= Logistic(data1,y_data1,data5[list(data1.columns)])
y_pred_d2= Logistic(data2,y_data2,data6[list(data2.columns)])
y_pred_d3= Logistic(data3,y_data3,data7[list(data3.columns)])
y_pred_d4= Logistic(data4,y_data4,data8[list(data4.columns)])
y_pred_d=np.hstack([y_pred_d1,y_pred_d2,y_pred_d3,y_pred_d4])
export(ids_test,y_pred_d,"year_tot")

#escalonando com todas as variáveis
#normalização
df_norm1=norm(data1)
y_norm_d1=Logistic(df_norm1,y_data1,data5[list(df_norm1.columns)])
df_norm2=norm(data2)
y_norm_d2=Logistic(df_norm2,y_data2,data6[list(df_norm2.columns)])
df_norm3=norm(data3)
y_norm_d3=Logistic(df_norm3,y_data3,data7[list(df_norm3.columns)])
df_norm4=norm(data4)
y_norm_d4=Logistic(df_norm4,y_data4,data8[list(df_norm4.columns)])
y_pred_dnorm=np.hstack([y_norm_d1,y_norm_d2,y_norm_d3,y_norm_d4])
export(ids_test,y_pred_dnorm,"year_tot_norm")

#min max
df_min_max1=min_max(data1)
y_mm_d1=Logistic(df_min_max1,y_data1,data5[list(df_min_max1.columns)])
df_min_max2=min_max(data2)
y_mm_d2=Logistic(df_min_max2,y_data2,data6[list(df_min_max2.columns)])
df_min_max3=min_max(data3)
y_mm_d3=Logistic(df_min_max3,y_data3,data7[list(df_min_max3.columns)])
df_min_max4=min_max(data4)
y_mm_d4=Logistic(df_min_max4,y_data4,data8[list(df_min_max4.columns)])
y_pred_dmm=np.hstack([y_mm_d1,y_mm_d2,y_mm_d3,y_mm_d4])
export(ids_test,y_pred_dmm,"year_tot_minmax")

#AMOSTRA BALANCEADA
import random
random.seed(10)
l=(y_train==1)
index=[i for i, x in enumerate(l) if x]
pos=df_train.iloc[index,:]
pos_y=y_train.iloc[index]
mask = np.ones(len(df_train), np.bool)
mask[index] = 0
neg = df_train[mask]
aleat=neg.iloc[np.random.choice(np.arange(len(neg)), 551, False)]
rand_index=list(aleat.index.values)
aleat_y= y_train.loc[rand_index]
frames_x=[pos,aleat]
frames_y=[pos_y,aleat_y]
df_train_bal=pd.concat(frames_x)
y_train_bal=pd.concat(frames_y)

#ALL FEATURES
y_pred_bal=Logistic(df_train_bal,y_train_bal,df_test[list(df_train_bal.columns)])
export(ids_test,y_pred_bal,"bal_base_tot")

#logistica normalizada
df_norm_bal=norm(df_train_bal)
y_pred_bal_norm= Logistic(df_norm_bal,y_train_bal,df_test[list(df_norm_bal.columns)])
export(ids_test,y_pred_bal_norm,"bal_norm_base_tot")

#logistica min max
df_mm_bal=min_max(df_train_bal)
y_pred_bal_mm=Logistic(df_mm_bal,y_train_bal,df_test[list(df_mm_bal.columns)])
export(ids_test,y_pred_bal_mm,"bal_mm_base_tot")

#selecionando variaveis
#RFE
df_sel_train_bal=sel_rfe(df_train_bal, y_train_bal,30)
y_sel_bal=Logistic(df_sel_train_bal,y_train_bal,df_test[list(df_sel_train_bal.columns)])
export(ids_test,y_sel_bal,"bal_sel_feat")
df_sel_train2_bal=sel_rfe(df_train_bal, y_train_bal, 20)
y_sel_bal2=Logistic(df_sel_train2_bal,y_train_bal,df_test[list(df_sel_train2_bal.columns)])
export(ids_test,y_sel_bal,"bal_sel_feat2")
df_sel_train3_bal=sel_rfe(df_train_bal, y_train_bal, 10)
y_sel_bal3=Logistic(df_sel_train3_bal,y_train_bal,df_test[list(df_sel_train3_bal.columns)])
export(ids_test,y_sel_bal,"bal_sel_feat3")

#TRESS
df_sel_train10_bal=feat_imp(df_train_bal, y_train_bal,200)
y_sel_bal10=Logistic(df_sel_train10_bal,y_train_bal,df_test[list(df_sel_train10_bal.columns)])
export(ids_test,y_sel_bal10,"bal_sel_feat10")
df_sel_train8_bal=feat_imp(df_train_bal, y_train_bal,150)
y_sel_bal8=Logistic(df_sel_train8_bal,y_train_bal,df_test[list(df_sel_train8_bal.columns)])
export(ids_test,y_sel_bal8,"bal_sel_feat8")
df_sel_train9_bal=feat_imp(df_train_bal, y_train_bal,100)
y_sel_bal9=Logistic(df_sel_train9_bal,y_train_bal,df_test[list(df_sel_train9_bal.columns)])
export(ids_test,y_sel_bal9,"bal_sel_feat9")
df_sel_train7_bal=feat_imp(df_train_bal, y_train_bal,50)
y_sel_bal7=Logistic(df_sel_train7_bal,y_train_bal,df_test[list(df_sel_train7_bal.columns)])
export(ids_test,y_sel_bal7,"bal_sel_feat7")
df_sel_train4_bal=feat_imp(df_train_bal, y_train_bal,30)
y_sel_bal4=Logistic(df_sel_train4_bal,y_train_bal,df_test[list(df_sel_train4_bal.columns)])
export(ids_test,y_sel_bal4,"bal_sel_feat4")
df_sel_train5_bal=feat_imp(df_train_bal, y_train_bal, 20)
y_sel_bal5=Logistic(df_sel_train5_bal,y_train_bal,df_test[list(df_sel_train5_bal.columns)])
export(ids_test,y_sel_bal5,"bal_sel_feat5")
df_sel_train6_bal=feat_imp(df_train_bal, y_train_bal, 10)
y_sel_bal6=Logistic(df_sel_train6_bal,y_train_bal,df_test[list(df_sel_train6_bal.columns)])
export(ids_test,y_sel_bal6,"bal_sel_feat6")

#ANALISE ANOS
#todas as variáveis
y_bal_2008=Logistic(df_train_bal[df_train_bal["Year"]==2007],y_train_bal[df_train_bal[df_train_bal["Year"]==2007].index.values],data5[df_train_bal[df_train_bal["Year"]==2007].columns])
y_bal_2010=Logistic(df_train_bal[df_train_bal["Year"]==2009],y_train_bal[df_train_bal[df_train_bal["Year"]==2009].index.values],data6[df_train_bal[df_train_bal["Year"]==2009].columns])
y_bal_2012=Logistic(df_train_bal[df_train_bal["Year"]==2011],y_train_bal[df_train_bal[df_train_bal["Year"]==2011].index.values],data7[df_train_bal[df_train_bal["Year"]==2011].columns])
y_bal_2014=Logistic(df_train_bal[df_train_bal["Year"]==2013],y_train_bal[df_train_bal[df_train_bal["Year"]==2013].index.values],data8[df_train_bal[df_train_bal["Year"]==2013].columns])
y_bal_year=np.hstack([y_bal_2008,y_bal_2010,y_bal_2012,y_bal_2014])
export(ids_test,y_bal_year,"bal_year")

#escalonando com todas as variáveis
#normalização
df_norm1_bal=norm(df_train_bal[df_train_bal["Year"]==2007])
y_bal_norm_2008=Logistic(df_norm1_bal,y_train_bal[df_train_bal[df_train_bal["Year"]==2007].index.values],data5[list(df_norm1_bal.columns)])
df_norm2_bal=norm(df_train_bal[df_train_bal["Year"]==2009])
y_bal_norm_2010=Logistic(df_norm2_bal,y_train_bal[df_train_bal[df_train_bal["Year"]==2009].index.values],data6[list(df_norm2_bal.columns)])
df_norm3_bal=norm(df_train_bal[df_train_bal["Year"]==2011])
y_bal_norm_2012=Logistic(df_norm3_bal,y_train_bal[df_train_bal[df_train_bal["Year"]==2011].index.values],data7[list(df_norm3_bal.columns)])
df_norm4_bal=norm(df_train_bal[df_train_bal["Year"]==2013])
y_bal_norm_2014=Logistic(df_norm4_bal,y_train_bal[df_train_bal[df_train_bal["Year"]==2013].index.values],data8[list(df_norm4_bal.columns)])
y_bal_norm_year=np.hstack([y_bal_norm_2008,y_bal_norm_2010,y_bal_norm_2012,y_bal_norm_2014])
export(ids_test,y_bal_norm_year,"bal_norm_year")

#min max
df_min_max1_bal=min_max(df_train_bal[df_train_bal["Year"]==2007])
y_bal_min_max_2008=Logistic(df_min_max1_bal,y_train_bal[df_train_bal[df_train_bal["Year"]==2007].index.values],data5[list(df_min_max1_bal.columns)])
df_min_max2_bal=min_max(df_train_bal[df_train_bal["Year"]==2009])
y_bal_min_max_2010=Logistic(df_min_max2_bal,y_train_bal[df_train_bal[df_train_bal["Year"]==2009].index.values],data6[list(df_min_max2_bal.columns)])
df_min_max3_bal=min_max(df_train_bal[df_train_bal["Year"]==2011])
y_bal_min_max_2012=Logistic(df_min_max3_bal,y_train_bal[df_train_bal[df_train_bal["Year"]==2011].index.values],data7[list(df_min_max3_bal.columns)])
df_min_max4_bal=min_max(df_train_bal[df_train_bal["Year"]==2013])
y_bal_min_max_2014=Logistic(df_min_max4_bal,y_train_bal[df_train_bal[df_train_bal["Year"]==2013].index.values],data8[list(df_min_max4_bal.columns)])
y_bal_min_max_year=np.hstack([y_bal_min_max_2008,y_bal_min_max_2010,y_bal_min_max_2012,y_bal_min_max_2014])
export(ids_test,y_bal_min_max_year,"bal_min_max_year")



#####RESPONDENDO AS PERGUNTAS
import matplotlib.pyplot as plt
df_sel_train17=feat_imp(df_train, y_train,200)
teste=df_test[list(df_sel_train17.columns)]
y_pred_sel_train6=Logistic(df_sel_train17,y_train,teste)
export(ids_test,y_pred_sel_train6,"sel_train7")

classifier = LogisticRegression()  
classifier.fit(df_sel_train17,y_train)
#y_pred_train = classifier.predict(x_train)
y_pred_train = classifier.predict_proba(df_sel_train17)[:,1]  
y_pred_trainf =getNormalizedTarget(y_pred_train)
index_prev= np.where(y_pred_trainf==1)
x=df_sel_train17.iloc[index_prev]


x.to_csv("analisys.csv", index=False)























