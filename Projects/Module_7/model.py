import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import os

current_path = os.getcwd()
data = pd.read_csv(f"{current_path}/data/2015.csv")
for datapoint in data:
    pd.assign

### Feature modification
def feature_mod(data):
  

features = ['Country','Region','Happiness Rank','Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Generosity','Dystopia Residual']
# Features describe how important the citizens of the country think each feature is to improving overall happiness in a country. 

### Model prep
X = pd.get_dummies(data[features], drop_first=True)
y = data["users"]

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