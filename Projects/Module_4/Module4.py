import pandas as pd
from lets_plot import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time

bikes = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv')

### Feature modification
def feature_mod(data):
  data['month'] = pd.DatetimeIndex(data['dteday']).month
  data['year'] = pd.DatetimeIndex(data['dteday']).year
  data.drop(columns=["dteday"], inplace=True)
  data.drop(columns=["hum"], inplace=True)

  return data

bikes = feature_mod(bikes)
bikes["users"] = bikes["casual"] + bikes["registered"]
bikes.drop(columns=["casual", "registered"], inplace=True)
bikes.filter(like="2011", axis=0)
bikes.filter(like="2020", axis=0)

features = ['month', 'year', "hr", "temp_c", "feels_like_c", "windspeed", "weathersit", "season", "holiday", "workingday"]


### Model prep
X = pd.get_dummies(bikes[features], drop_first=True)
y = bikes["users"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

norm = MinMaxScaler().fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)
early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=30)


### Model
dnn_model = keras.Sequential()
dnn_model.add(Dense(128, input_dim=len(X_train[0]), activation='relu'))
dnn_model.add(Dropout(.5))
dnn_model.add(Dense(256, activation='relu'))
dnn_model.add(Dense(64, activation='leaky_relu'))
dnn_model.add(Dense(1, activation='relu'))

opt = keras.optimizers.Adam()
dnn_model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mse'])
dnn_model.summary()

start = time.time()

dnn_model.fit(
  X_train,
  y_train,
  validation_split=0.2,
  verbose=0, 
  epochs=20,
  batch_size=20, 
  callbacks=[early_stop],
  shuffle=False)

end = time.time()
print(f"Training time: {end - start:.2f} seconds")


### Test
y_pred = dnn_model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")


### Mini holdout
bikes = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/biking_holdout_test_mini.csv')
bikes = feature_mod(bikes)

test_X = pd.get_dummies(bikes[features], drop_first=True)
test_X = test_X.reindex(columns=X.columns, fill_value=0)
test_X = norm.transform(test_X)

y_pred = np.round(dnn_model.predict(test_X),1)
my_predictions = pd.DataFrame(y_pred, columns=['predictions'])
my_predictions.to_csv(path_or_buf="Projects/Module_4/team5-module4-predictions.csv", index=False)


### December
bikes = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes_december.csv")
bikes['day'] = pd.DatetimeIndex(bikes['dteday']).day
bikes = feature_mod(bikes)

test_X = pd.get_dummies(bikes[features], drop_first=True)
test_X = test_X.reindex(columns=X.columns, fill_value=0)
test_X = norm.transform(test_X)

y_pred = np.round(dnn_model.predict(test_X),1)
my_predictions = pd.DataFrame({
    'day': bikes['day'].values,
    'month': bikes['month'].values,
    'year': bikes['year'].values,
    'hour': bikes['hr'].values,
    'predictions': y_pred.flatten
})
my_predictions.to_csv(path_or_buf="Projects/Module_4/team5-module4-predictions-december.csv", index=False)