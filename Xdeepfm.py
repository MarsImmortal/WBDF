import pandas as pd
data = pd.read_csv('nursery.csv')

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.models import xDeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

sparse_features = data.columns[:-1]
dense_features = []

data[sparse_features] = data[sparse_features].fillna('-1', )
# data[dense_features] = data[dense_features].fillna(0,)
target = data.columns[-1]

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

data[target] = LabelEncoder().fit_transform(data[target])

# mms = MinMaxScaler(feature_range=(0,1))
# data[dense_features] = mms.fit_transform(data[dense_features])

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                          for i, feat in enumerate(sparse_features)]
# + [DenseFeat(feat, 1,) for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

train, test = train_test_split(data, test_size=0.2)

train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}

model = xDeepFM(linear_feature_columns,dnn_feature_columns,task='binary')
model.compile("adam", "binary_crossentropy",
              metrics=['accuracy'], )

history = model.fit(train_model_input, train[[target]].values,
                    batch_size=256, epochs=200, verbose=2, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=256)