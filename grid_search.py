import numpy as np
import pandas as pd
import time
from yearly_data import YearlyData
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

def getNormalizedTarget(array_target):
	for index in range(len(array_target)):
		array_target[index] = 1 if array_target[index] >= 0.9 else 0
		# array_target[index] = int(array_target[index])

	return array_target


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



#SVM
# -----------------------------------------------------------------------------
# print("\n\nTraining model SVM...")
# svm = SVC()
# svm.fit(train_features, train_target)

# print("Predicting...")
# pred = svm.predict(validation_features)
# pred = getNormalizedTarget(pred)

# svm_accuracy = accuracy_score(validation_target, pred)
# print('Accuracy for SVM:', svm_accuracy)
# -----------------------------------------------------------------------------


#Random Forest Regressor
# -----------------------------------------------------------------------------
print("\n\nTraining model Random Forest Regressor...")
rf = RandomForestClassifier()

params = {'n_estimators':[10,15,20,25,30],
          'min_samples_leaf':[1,2,3],
          'min_samples_split':[3,4,5,6,7]}

grid_search = GridSearchCV(rf, param_grid=params, n_jobs=-1)

grid_search.fit(train_features, train_target)

print("Predicting...")
predictions = grid_search.predict(validation_features)
predictions = getNormalizedTarget(predictions)

accuracy = accuracy_score(validation_target, predictions)
print('Accuracy for random forest:', accuracy)
print("Best Hyper Parameters:", grid_search.best_params_)
# -----------------------------------------------------------------------------


#Gradient Tree Boosting
# -----------------------------------------------------------------------------
# print("\n\nTraining model Gradient Tree Boosting...")
# gtb = GradientBoostingClassifier(n_estimators=1000)
# gtb.fit(train_features, train_target)

# print("Predicting...")
# pred_gtb = gtb.predict(validation_features)
# pred_gtb = getNormalizedTarget(pred_gtb)

# gtb_accuracy = accuracy_score(validation_target, pred_gtb)
# print('Accuracy for Gradient Tree Boosting:', gtb_accuracy)
# -----------------------------------------------------------------------------
print("Predicting test")
sub_data_test = data_test.loc[:,:]
ids_test = np.array(sub_data_test['Id'])
sub_data_test = sub_data_test.drop('Id', axis = 1)
test_features = np.array(sub_data_test)

# test_SVM = svm.predict(test_features)
test_RF = grid_search.predict(test_features)
# test_GTB = gtb.predict(test_features)

df_result = pd.DataFrame(ids_test)
# df_result.insert(loc=1, column='WnvPresent', value=test_SVM)
# df_result.to_csv("svm_prediction.csv", index=False)

# df_result = df_result.drop(['WnvPresent'], axis=1)

df_result.insert(loc=1, column='WnvPresent', value=test_RF)
df_result.to_csv("grid_search{}.csv".format(int(time.time())), index=False)

# df_result = df_result.drop(['WnvPresent'], axis=1)

# df_result.insert(loc=1, column='WnvPresent', value=test_GTB)
# df_result.to_csv("gradient_tree_boosting.csv", index=False)