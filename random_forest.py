import os
import time
import numpy as np
import pandas as pd
from yearly_data import YearlyData
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

def getNormalizedTarget(array_target):
	for index in range(len(array_target)):
		array_target[index] = 1 if array_target[index] >= 0.9 else 0

	return array_target

print("Reading datasets...")
data = pd.read_csv("features/train.csv")
data_test = pd.read_csv("features/test.csv")

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
# model_name = "support_vector_machines"
# model = SVC()
# model.fit(train_features, train_target)

# print("Predicting...")
# pred = model.predict(validation_features)
# pred = getNormalizedTarget(pred)

# accuracy = metrics.accuracy_score(validation_target, pred)
# print('Accuracy for SVM:', accuracy)
# -----------------------------------------------------------------------------


#Random Forest Regressor
# -----------------------------------------------------------------------------
print("\n\nTraining model Random Forest Regressor...")
model_name = "random_forest"

model = RandomForestRegressor(n_estimators=10, max_depth=1)
model.fit(train_features, train_target)

print("Predicting...")
predictions = model.predict(validation_features)
predictions = getNormalizedTarget(predictions)

accuracy = metrics.accuracy_score(validation_target, predictions)
print('Accuracy for random forest:', accuracy)
# -----------------------------------------------------------------------------


#Random Forest Classifier
# -----------------------------------------------------------------------------
# print("\n\nTraining model Random Forest Classifier...")
# model_name = "random_forest_classifier"

# model = RandomForestClassifier(n_jobs=-1, n_estimators=1000)
# model.fit(train_features, train_target)

# print("Predicting...")
# predictions = model.predict(validation_features)
# predictions = getNormalizedTarget(predictions)

# accuracy = metrics.accuracy_score(validation_target, predictions)
# print('Accuracy for random forest:', accuracy)
# -----------------------------------------------------------------------------

#Logistic Regression
# -----------------------------------------------------------------------------
# print("\n\nTraining model Random Forest Regressor...")
# model_name = "logistic_regression"

# model = LogisticRegression()  
# model.fit(train_features, train_target)

# print("Predicting...")
# predictions = model.predict_proba(validation_features)[:,1]
# predictions = getNormalizedTarget(predictions)

# accuracy = metrics.accuracy_score(validation_target, predictions)
# print('Accuracy for logistic regression:', accuracy)
# -----------------------------------------------------------------------------


#Gradient Tree Boosting
# -----------------------------------------------------------------------------
# print("\n\nTraining model Gradient Tree Boosting...")
# model_name = "gradient_tree_boosting"

# param_test1 = {'n_estimators': [5,16,2], 'max_depth': [5,16,2], 'min_samples_split': [200,1001,200], 'subsample': [0.6,0.7,0.75,0.8,0.85,0.9]}
# estimator = GradientBoostingClassifier(learning_rate=0.035, min_samples_leaf=50, max_features='sqrt', random_state=10)

# model = GridSearchCV(estimator = estimator, param_grid = param_test1, scoring='roc_auc',n_jobs=2,iid=False, cv=5)
# model.fit(train_features, train_target)

# print("Predicting...")
# pred_model = model.predict(validation_features)
# pred_model = getNormalizedTarget(pred_model)

# accuracy = metrics.accuracy_score(validation_target, pred_model)
# print('Accuracy for Gradient Tree Boosting:', accuracy)
# -----------------------------------------------------------------------------

print("Predicting test")

sub_data_test = data_test.loc[:,:]
ids_test = np.array(sub_data_test['Id'])
sub_data_test = sub_data_test.drop('Id', axis = 1)
test_features = np.array(sub_data_test)

test_model = model.predict(test_features)

prediction_directory = "predictions"

if not os.path.exists(prediction_directory):
    os.makedirs(prediction_directory)

df_result = pd.DataFrame(ids_test)
df_result.insert(loc=1, column='WnvPresent', value=test_model)
df_result.to_csv(prediction_directory + "/" + model_name + "_{}.csv".format(int(time.time())), index=False)


