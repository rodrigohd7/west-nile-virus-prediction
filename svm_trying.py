import numpy as np
import pandas as pd
from yearly_data import YearlyData
from sklearn import ensemble, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

def getNormalizedTarget(array_target):
	for index in range(len(array_target)):
		array_target[index] = 1 if array_target[index] >= np.mean(array_target)  else 0
		# array_target[index] = int(array_target[index])

	return array_target

def export(ids_test,y_pred_test1,name):
    df_result = pd.DataFrame()
    df_result.insert(loc=0, column='Id', value=ids_test)
    df_result.insert(loc=1, column='WnvPresent', value=y_pred_test1)
    df_result.to_csv(name+".csv", index=False)
    

yearly = YearlyData("./train.csv", "./weather.csv", test_csv="./test.csv")
yearly_test = YearlyData("./test.csv","./weather.csv")

print("Pre-processing datasets...")
data = yearly.process()
data_test = yearly_test.process()

for x in range(2007,2014):
	if (x % 2 == 1):
		print("Processing for year {}".format(x))

		sub_data = data.loc[data['Year'] == x]
		
		target = np.array(sub_data['WnvPresent'])
		
		sub_data = sub_data.drop('WnvPresent', axis = 1)
		sub_data = sub_data.drop('NumMosquitos', axis = 1)
		
		features = np.array(sub_data)

		features_size = len(features)

		train_features = features[:int(features_size * 0.7), :]
		validation_features = features[len(train_features) : features_size, :]

		train_target = target[:int(features_size * 0.7)]
		validation_target = target[len(train_features) : features_size]



#SVM
# -----------------------------------------------------------------------------
y_data=data["WnvPresent"]
df_train = data.drop(["WnvPresent", "NumMosquitos"],axis=1)      
ids_test = np.array(data_test['Id'])        


print("\n\nTraining model SVM...")
svm = SVC(probability=True)

#Hyper Parameters Set
params = {'C': [6,7,8,9,10,11,12], 
          'kernel': ['linear','rbf']}
#Making models with hyper parameters sets
#best parameters C=10 with linear kernel
model1 = GridSearchCV(svm, param_grid=params, n_jobs=-1)
model1.fit(df_train, y_data)
print("Best Hyper Parameters:",model1.best_params_)
prediction=model1.predict_proba(data_test[df_train.columns])[:,1]
prediction2=model1.predict_proba(df_train)[:,1]
predict2=getNormalizedTarget(prediction2)
confusion_matrix=pd.crosstab(y_data, predict2, rownames=['True'], colnames=['Predicted'], margins=True)
print("Treino \n", confusion_matrix)
print("Treino \n", accuracy_score(y_data,predict2))
print("Treino \n", classification_report(y_data,predict2))
export(ids_test,prediction,"SVM_hyper_params2")

# -----------------------------------------------------------------------------

svm2=SVC(C=6,kernel="linear",probability=True)
svm2.fit(df_train, y_data)
prediction=svm2.predict_proba(data_test[df_train.columns])[:,1]
prediction2=svm2.predict_proba(df_train)[:,1]
predict2=getNormalizedTarget(prediction2)
confusion_matrix=pd.crosstab(y_data, predict2, rownames=['True'], colnames=['Predicted'], margins=True)
print("Treino \n", confusion_matrix)
print("Treino \n", accuracy_score(y_data,predict2))
print("Treino \n", classification_report(y_data,predict2))
export(ids_test,prediction,"SVM_hyper_6_linear")


svm2=SVC(C=7,kernel="linear",probability=True)
svm2.fit(df_train, y_data)
prediction=svm2.predict_proba(data_test[df_train.columns])[:,1]
prediction2=svm2.predict_proba(df_train)[:,1]
predict2=getNormalizedTarget(prediction2)
confusion_matrix=pd.crosstab(y_data, predict2, rownames=['True'], colnames=['Predicted'], margins=True)
print("Treino \n", confusion_matrix)
print("Treino \n", accuracy_score(y_data,predict2))
print("Treino \n", classification_report(y_data,predict2))
export(ids_test,prediction,"SVM_hyper_7_linear")

svm2=SVC(C=8,kernel="linear",probability=True)
svm2.fit(df_train, y_data)
prediction=svm2.predict_proba(data_test[df_train.columns])[:,1]
prediction2=svm2.predict_proba(df_train)[:,1]
predict2=getNormalizedTarget(prediction2)
confusion_matrix=pd.crosstab(y_data, predict2, rownames=['True'], colnames=['Predicted'], margins=True)
print("Treino \n", confusion_matrix)
print("Treino \n", accuracy_score(y_data,predict2))
print("Treino \n", classification_report(y_data,predict2))
export(ids_test,prediction,"SVM_hyper_8_linear")

svm2=SVC(C=9,kernel="linear",probability=True)
svm2.fit(df_train, y_data)
prediction=svm2.predict_proba(data_test[df_train.columns])[:,1]
prediction2=svm2.predict_proba(df_train)[:,1]
predict2=getNormalizedTarget(prediction2)
confusion_matrix=pd.crosstab(y_data, predict2, rownames=['True'], colnames=['Predicted'], margins=True)
print("Treino \n", confusion_matrix)
print("Treino \n", accuracy_score(y_data,predict2))
print("Treino \n", classification_report(y_data,predict2))
export(ids_test,prediction,"SVM_hyper_9_linear")


svm2=SVC(C=11,kernel="linear",probability=True)
svm2.fit(df_train, y_data)
prediction=svm2.predict_proba(data_test[df_train.columns])[:,1]
prediction2=svm2.predict_proba(df_train)[:,1]
predict2=getNormalizedTarget(prediction2)
confusion_matrix=pd.crosstab(y_data, predict2, rownames=['True'], colnames=['Predicted'], margins=True)
print("Treino \n", confusion_matrix)
print("Treino \n", accuracy_score(y_data,predict2))
print("Treino \n", classification_report(y_data,predict2))
export(ids_test,prediction,"SVM_hyper_11_linear")


svm2=SVC(C=12,kernel="linear",probability=True)
svm2.fit(df_train, y_data)
prediction=svm2.predict_proba(data_test[df_train.columns])[:,1]
prediction2=svm2.predict_proba(df_train)[:,1]
predict2=getNormalizedTarget(prediction2)
confusion_matrix=pd.crosstab(y_data, predict2, rownames=['True'], colnames=['Predicted'], margins=True)
print("Treino \n", confusion_matrix)
print("Treino \n", accuracy_score(y_data,predict2))
print("Treino \n", classification_report(y_data,predict2))
export(ids_test,prediction,"SVM_hyper_12_linear")

svm2=SVC(C=6,kernel="rbf",probability=True)
svm2.fit(df_train, y_data)
prediction=svm2.predict_proba(data_test[df_train.columns])[:,1]
prediction2=svm2.predict_proba(df_train)[:,1]
predict2=getNormalizedTarget(prediction2)
confusion_matrix=pd.crosstab(y_data, predict2, rownames=['True'], colnames=['Predicted'], margins=True)
print("Treino \n", confusion_matrix)
print("Treino \n", accuracy_score(y_data,predict2))
print("Treino \n", classification_report(y_data,predict2))
export(ids_test,prediction,"SVM_hyper_6_rbf")


svm2=SVC(C=7,kernel="rbf",probability=True)
svm2.fit(df_train, y_data)
prediction=svm2.predict_proba(data_test[df_train.columns])[:,1]
prediction2=svm2.predict_proba(df_train)[:,1]
predict2=getNormalizedTarget(prediction2)
confusion_matrix=pd.crosstab(y_data, predict2, rownames=['True'], colnames=['Predicted'], margins=True)
print("Treino \n", confusion_matrix)
print("Treino \n", accuracy_score(y_data,predict2))
print("Treino \n", classification_report(y_data,predict2))
export(ids_test,prediction,"SVM_hyper_7_rbf")

svm2=SVC(C=8,kernel="rbf",probability=True)
svm2.fit(df_train, y_data)
prediction=svm2.predict_proba(data_test[df_train.columns])[:,1]
prediction2=svm2.predict_proba(df_train)[:,1]
predict2=getNormalizedTarget(prediction2)
confusion_matrix=pd.crosstab(y_data, predict2, rownames=['True'], colnames=['Predicted'], margins=True)
print("Treino \n", confusion_matrix)
print("Treino \n", accuracy_score(y_data,predict2))
print("Treino \n", classification_report(y_data,predict2))
export(ids_test,prediction,"SVM_hyper_8_rbf")

svm2=SVC(C=9,kernel="rbf",probability=True)
svm2.fit(df_train, y_data)
prediction=svm2.predict_proba(data_test[df_train.columns])[:,1]
prediction2=svm2.predict_proba(df_train)[:,1]
predict2=getNormalizedTarget(prediction2)
confusion_matrix=pd.crosstab(y_data, predict2, rownames=['True'], colnames=['Predicted'], margins=True)
print("Treino \n", confusion_matrix)
print("Treino \n", accuracy_score(y_data,predict2))
print("Treino \n", classification_report(y_data,predict2))
export(ids_test,prediction,"SVM_hyper_9_rbf")


svm2=SVC(C=11,kernel="rbf",probability=True)
svm2.fit(df_train, y_data)
prediction=svm2.predict_proba(data_test[df_train.columns])[:,1]
prediction2=svm2.predict_proba(df_train)[:,1]
predict2=getNormalizedTarget(prediction2)
confusion_matrix=pd.crosstab(y_data, predict2, rownames=['True'], colnames=['Predicted'], margins=True)
print("Treino \n", confusion_matrix)
print("Treino \n", accuracy_score(y_data,predict2))
print("Treino \n", classification_report(y_data,predict2))
export(ids_test,prediction,"SVM_hyper_11_rbf")


svm2=SVC(C=12,kernel="rbf",probability=True)
svm2.fit(df_train, y_data)
prediction=svm2.predict_proba(data_test[df_train.columns])[:,1]
prediction2=svm2.predict_proba(df_train)[:,1]
predict2=getNormalizedTarget(prediction2)
confusion_matrix=pd.crosstab(y_data, predict2, rownames=['True'], colnames=['Predicted'], margins=True)
print("Treino \n", confusion_matrix)
print("Treino \n", accuracy_score(y_data,predict2))
print("Treino \n", classification_report(y_data,predict2))
export(ids_test,prediction,"SVM_hyper_12_rbf")