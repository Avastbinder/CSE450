import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

data = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv")

data['datetime'] = data['date'].str.split('T').str[0]
# Extract year, month, day from compact YYYYMMDD format
data = data.assign(
    year = data['datetime'].str[0:4].astype(int),
    month = data['datetime'].str[4:6].astype(int),
    day = data['datetime'].str[6:8].astype(int)
)
# Drop the intermediate 'date' column
data = data.drop(columns=['date'])
data = data.drop(columns=['datetime'])

# Model features
features = ["year","month","day","bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","grade","sqft_above","sqft_basement","yr_built","yr_renovated","zipcode","lat","long","sqft_living15","sqft_lot15"]

X = pd.get_dummies(data[features], drop_first=True)
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model parameters
model = XGBRegressor(
    max_depth = 3,
    reg_lambda = 3,
    reg_alpha = 2,
    random_state = 42,
    colsample_bytree = 0.5
)

# Run model
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Print R2 score and RMSE score
r2 = r2_score(y_test, predictions)
result = root_mean_squared_error(y_test, predictions)
print(f"R² Score: {r2:.4f}")
print(f"RMSE Score: {result:.4f}")

### Test model with mini holdout
data = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini.csv")

data['datetime'] = data['date'].str.split('T').str[0]
#Extract year, month, day from compact YYYYMMDD format
data = data.assign(
    year = data['datetime'].str[0:4].astype(int),
    month = data['datetime'].str[4:6].astype(int),
    day = data['datetime'].str[6:8].astype(int)
)
# Drop the intermediate 'date' column
data = data.drop(columns=['date'])
data = data.drop(columns=['datetime'])

# Match training features
test_X = pd.get_dummies(data[features], drop_first=True)
test_X = test_X.reindex(columns=X.columns, fill_value=0)

# Run model on mini holdout and send results to a CSV file
predictions = model.predict(test_X)
my_predictions = pd.DataFrame(predictions, columns=['predictions'])
my_predictions.to_csv(path_or_buf="Projects/Module_3/team3-module3-predictions.csv", index=False)