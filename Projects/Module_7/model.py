import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path

data_folder = Path(__file__).parent
data = pd.read_csv(f"{data_folder}/data/data.csv")

features = ['Country','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Generosity','year']
# Features describe how important the citizens of the country think each feature is to improving overall happiness in a country. 

### Model prep
X = pd.get_dummies(data[features], drop_first=True)
y = data["Happiness Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


### Model
# XGBoost model parameters
model = XGBRegressor(
    eta = 0.2,
    max_depth = 0,
    reg_lambda = 7,
    reg_alpha = 1,
    colsample_bytree = 0.6,
    objective = 'reg:tweedie',
    seed = 42
)

# Run model
model.fit(X_train, y_train)
predictions = model.predict(X_test)


### Test
y_pred = model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nInternal test statistics:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")


### holdout
holdout = pd.read_csv(f"{data_folder}/data/2019_holdout.csv")
test_holdout = pd.read_csv(f"{data_folder}/data/2019_actual.csv")

test_X = pd.get_dummies(holdout[features], drop_first=True)
test_X = test_X.reindex(columns=X.columns, fill_value=0)

y_pred = np.round(model.predict(test_X),3)
my_predictions = pd.DataFrame(y_pred, columns=['predictions'])
my_predictions.to_csv(path_or_buf=data_folder / f"data/predictions.csv", index=False)

rmse = root_mean_squared_error(test_holdout['Happiness Score'], y_pred)
r2 = r2_score(test_holdout['Happiness Score'], y_pred)

print("\nHoldout test statistics:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}\n")