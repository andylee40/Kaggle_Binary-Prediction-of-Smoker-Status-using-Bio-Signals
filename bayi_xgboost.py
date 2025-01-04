import numpy as np 
import pandas as pd 
import seaborn as sns
import io
import requests
import re
import warnings
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

train=pd.read_csv('./train.csv')
test=pd.read_csv('./test.csv')

train=train.set_index('id')
test=test.set_index('id')

y=train.iloc[:,-1]
x=train.iloc[:,:-1]

train_ana=x.copy()
test_ana=test.copy()

train_ana['label']='train'
test_ana['label']='test'
all_data=pd.concat([train_ana,test_ana])

#特徵工程
model_data=all_data.copy()
le = LabelEncoder()
model_data['age'] = le.fit_transform(model_data['age'])
model_data['height(cm)'] = le.fit_transform(model_data['height(cm)'])
model_data['weight(kg)'] = le.fit_transform(model_data['weight(kg)'])
model_data['waist(cm)'] = le.fit_transform(model_data['waist(cm)'])
model_data['eyesight(left)'] = le.fit_transform(model_data['eyesight(left)'])
model_data['eyesight(right)'] = le.fit_transform(model_data['eyesight(right)'])
model_data['systolic'] = le.fit_transform(model_data['systolic'])
model_data['relaxation'] = le.fit_transform(model_data['relaxation'])
model_data['fasting blood sugar'] = le.fit_transform(model_data['fasting blood sugar'])
model_data['Cholesterol'] = le.fit_transform(model_data['Cholesterol'])
model_data['triglyceride'] = le.fit_transform(model_data['triglyceride'])
model_data['HDL'] = le.fit_transform(model_data['HDL'])
model_data['LDL'] = le.fit_transform(model_data['LDL'])
model_data['hemoglobin'] = le.fit_transform(model_data['hemoglobin'])
model_data['Urine protein'] = le.fit_transform(model_data['Urine protein'])
model_data['serum creatinine'] = le.fit_transform(model_data['serum creatinine'])
model_data['AST'] = le.fit_transform(model_data['AST'])
model_data['ALT'] = le.fit_transform(model_data['ALT'])
model_data['Gtp'] = le.fit_transform(model_data['Gtp'])

model_train=model_data[model_data['label']=='train'].drop('label', axis=1)
model_test=model_data[model_data['label']=='test'].drop('label', axis=1)


# 超參數範圍
pbounds = {
    'learning_rate': (0.01, 0.1),
    'max_depth': (3, 10),
    'min_child_weight': (1, 5),
    'subsample': (0.5, 0.7),
    'colsample_bytree': (0.5, 0.7),
    'n_estimators': (100, 500),
    'gamma': (0, 1),
    'reg_alpha': (0, 1),
    'reg_lambda': (0.5, 5),
    'base_score': (0.2, 1)
}

def xgboost_hyper_param(learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, n_estimators, gamma, reg_alpha, reg_lambda, base_score):
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)

    clf = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        base_score=base_score,
        objective='binary:logistic',
        tree_method="hist",
        device="cuda")
    
    return np.mean(cross_val_score(clf, model_train, y, cv=5, scoring='accuracy'))
    # return np.mean(cross_val_score(clf, model_train, y, cv=5, scoring='roc_auc'))
    
# 開始優化找最佳參數
optimizer = BayesianOptimization(
    f=xgboost_hyper_param,
    pbounds=pbounds,
    random_state=1,
)

# 執行優化
optimizer.maximize(init_points=5, n_iter=10)

# 印出最佳結果、參數
print(optimizer.max)

# 儲存最佳參數
best_param = optimizer.max['params']
max_depth_p = round(best_param['max_depth'])
learning_rate_p = best_param['learning_rate']
n_estimators_p = round(best_param['n_estimators'])
gamma_p = best_param['gamma']
reg_alpha_p = best_param['reg_alpha']
reg_lambda_p = best_param['reg_lambda']
min_child_weight_p = round(best_param['min_child_weight'])
subsample_p = best_param['subsample']
colsample_bytree_p = best_param['colsample_bytree']
base_score_p = best_param['base_score']

# 最佳參數訓練模型
clf = XGBClassifier(
    max_depth=max_depth_p,
    learning_rate=learning_rate_p,
    n_estimators=n_estimators_p,
    gamma=gamma_p,
    reg_alpha=reg_alpha_p,
    reg_lambda=reg_lambda_p,
    min_child_weight=min_child_weight_p,
    subsample=subsample_p,
    colsample_bytree=colsample_bytree_p,
    base_score=base_score_p,
    objective='binary:logistic',
    tree_method="hist",
    device="cuda")

clf.fit(model_train, y)

# 使用最佳模型進行預測
predict = clf.predict_proba(model_test)[:, 1]

# 提交資料
sub = model_test.copy()
sub.reset_index(inplace=True)
sub['smoking'] = predict
sub = sub[['id', 'smoking']]
sub.to_csv('sub10.csv', index=False)
