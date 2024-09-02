import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder,LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/higgs-dm.csv')

from sklearn.utils import shuffle

data = shuffle(data)

data1 = data.iloc[0:50000]
train, test = train_test_split(data1, test_size=0.2)
train.to_csv('./data_generated/train_hig.csv',index=False)
test.to_csv('./data_generated/test_hig.csv',index=False)