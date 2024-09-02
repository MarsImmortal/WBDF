import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv('adult-dm.csv')
data_new = shuffle(df)
data_new.to_csv('shuffle_data.csv',index=False)