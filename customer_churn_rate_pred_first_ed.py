
#data preprocessing

#libs & dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('/content/Churn_Modelling.csv')

dataset.head()

dataset.shape

dataset.info()

dataset.select_dtypes(include='object').columns

dataset.select_dtypes(include=['int64', 'float64']).columns

len(dataset.select_dtypes(include=['int64', 'float64']).columns)

len(dataset.select_dtypes(include='object').columns)

#statistical summary
dataset.describe()

#dealing with missing data
dataset.isnull().values.any()

dataset.isnull().values.sum()

#encode the categorical data
dataset.select_dtypes(include='object').columns

dataset.head()

dataset = dataset.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

dataset.head()

dataset.select_dtypes(include='object').columns

dataset['Geography'].unique()

dataset['Gender'].unique()

dataset.groupby('Geography').mean(numeric_only=True)

dataset.groupby('Gender').mean(numeric_only=True)

#one hot encoding
dataset = pd.get_dummies(data=dataset, drop_first=True)

dataset.head()

#countplot
sns.countplot(dataset['Exited'])
plt.plot

#customers staying with the bank
(dataset.Exited == 0).sum()

#customers leaving the bank
(dataset.Exited == 1).sum()

#correlation matrix & heatmap
dataset_2 = dataset.drop(columns='Exited')

dataset_2.corrwith(dataset['Exited']).plot.bar(
    figsize=(16,9), title='Correlated with exited', rot=45, grid=True
)

corr = dataset.corr()

plt.figure(figsize=(16,9))
sns.heatmap(corr, annot=True)

#splitting the dataset into train/set

dataset.head()

#independant/ Matrix of features
x = dataset.drop(columns='Exited')

#target/ dependant variable
y = dataset['Exited']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train.shape

x_test.shape

y_train.shape

y_test.shape

#feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train

x_test

#building the model

#logistic regression
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(x_train, y_train)

y_pred = classifier_lr.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

results = pd.DataFrame([['Logistic Regression', acc, f1, prec, rec]],
                       columns=['Model', 'Accuracy', 'F1', 'Precision', 'Recall']
                       )

results

cm = confusion_matrix(y_test, y_pred)
print(cm)

#cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier_lr, X=x_train, y=y_train, cv=10) #to increase accuracy

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))

#random forest
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state=0)
classifier_rf.fit(x_train, y_train)

y_pred = classifier_rf.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest', acc, f1, prec, rec]],
                       columns=['Model', 'Accuracy', 'F1', 'Precision', 'Recall']
                       )

results = results._append(model_results, ignore_index=True)

results

#cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier_rf, X=x_train, y=y_train, cv=10) #to increase accuracy

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))

#XG boost classification
from xgboost import XGBClassifier
classifier_xgb = XGBClassifier()
classifier_xgb.fit(x_train, y_train)

y_pred = classifier_xgb.predict(x_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

model_results = pd.DataFrame([['XGboost classifier', acc, f1, prec, rec]],
                       columns=['Model', 'Accuracy', 'F1', 'Precision', 'Recall']
                       )

results = results._append(model_results, ignore_index=True)

results

cm = confusion_matrix(y_test, y_pred)
print(cm)

#cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier_xgb, X=x_train, y=y_train, cv=10) #to increase accuracy

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))

#randomized search to find the best params
from sklearn.model_selection import RandomizedSearchCV

parameters  = {
    'learning_rate':[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    'max_depth':[3, 4, 5, 6, 7, 8, 10, 12, 15],
    'min_child_weight':[1, 3, 5, 7],
    'gamma':[0.0, 0.1, 0.2, 0.3, 0.4],
    'colsmaple_bytree':[0.3, 0.4, 0.5, 0.7]
}

parameters

randomized_search = RandomizedSearchCV(estimator=classifier_xgb, param_distributions=parameters, n_iter=5, n_jobs=-1, scoring='roc_auc',cv=5, verbose=3)

randomized_search.fit(x_train, y_train)

randomized_search.best_estimator_

randomized_search.best_params_

randomized_search.best_score_

#final model = xg_boost classifier
#XG boost classification
from xgboost import XGBClassifier
classifier = XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, colsmaple_bytree=0.4, device=None,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, feature_types=None, gamma=0.4, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.05, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=5,
              max_leaves=None, min_child_weight=1,
              monotone_constraints=None, multi_strategy=None, n_estimators=None,
              n_jobs=None, num_parallel_tree=None)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

model_results = pd.DataFrame([['Final XGboost', acc, f1, prec, rec]],
                       columns=['Model', 'Accuracy', 'F1', 'Precision', 'Recall']
                       )

results = results._append(model_results, ignore_index=True)

results

cm = confusion_matrix(y_test, y_pred)
print(cm)

#cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10) #to increase accuracy

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))

#predicting a single observation

dataset.head()

single_obs = [[625,	45,	5,	12500.01,	1,	0,	1,	101348.88, False,	False, True]]

single_obs

classifier.predict(sc.transform(single_obs))













