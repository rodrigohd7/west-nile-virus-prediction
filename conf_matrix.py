from yearly_data import YearlyData
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import scipy.stats as sc

def Logistic2(x_train,y_train,x_test): 
    classifier = LogisticRegression(random_state=0)  
    classifier.fit(x_train,y_train)
    y_pred_train = classifier.predict_proba(x_train)[:,1]
    y_pred_class = classifier.predict(x_train)
    y_pred_test = classifier.predict_proba(x_test)[:,1]   
    confusion_matrix=pd.crosstab(y_train, y_pred_class, rownames=['True'], colnames=['Predicted'], margins=True)
    print("Treino \n", confusion_matrix)
    print("Treino \n", classifier.score(df_train,y_train))
    return y_pred_test


yearly = YearlyData('../dataset/input/train.csv', '../dataset/input/weather.csv', test_csv='../dataset/input/test.csv')
yearly_test = YearlyData('../dataset/input/test.csv', '../dataset/input/weather.csv')

print("Pre-processing datasets...")
data = yearly.process()
data_test = yearly_test.process()

sub_data = data.loc[:,:]

target = np.array(sub_data['WnvPresent'])

sub_data = sub_data.drop('WnvPresent', axis = 1)
sub_data = sub_data.drop('NumMosquitos', axis = 1)

features = np.array(sub_data)

features_size = len(features)

train_features = features[:int(features_size * 0.7), :]
validation_features = features[len(train_features) : features_size, :]

train_target = target[:int(features_size * 0.7)]
validation_target = target[len(train_features) : features_size]

sub_data_test = data_test.loc[:,:]
ids_test = np.array(sub_data_test['Id'])
sub_data_test = sub_data_test.drop('Id', axis = 1)
test_features = np.array(sub_data_test)

y_pred_test1=Logistic2(train_features,train_target,test_features)