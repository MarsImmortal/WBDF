import operator

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt

data = pd.read_csv('./data/adult-dm.csv')
sparse_features = data.columns[:-1]
target = data.columns[-1]

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

data[target] = LabelEncoder().fit_transform(data[target])
traindata, testdata = train_test_split(data, test_size=0.2)

traindata = pd.get_dummies(traindata, columns = traindata.columns.values[0:-1], prefix_sep='_', dummy_na=False, drop_first=False)
print(traindata)

train = traindata.values[:,0:-1]
y_train = traindata.values[:,-1]

rf = RandomForestRegressor(n_estimators=100)
rf.fit(train, y_train)

fea_importance = rf.feature_importances_
column = traindata.columns.values[0:-1]
column_importance = list(zip(column,fea_importance))
column_importance.sort(key=operator.itemgetter(1), reverse=True)
print(column_importance)