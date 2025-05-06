import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from lets_plot import *
LetsPlot.setup_html()

data = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv")
data = data.dropna() # Delete N/A values

# Parse relevent data
features = ["job", "default", "pdays", "poutcome"]
X = pd.get_dummies(data[features], drop_first=True)
y = data["y"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

# Test with mini holdout
test = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank_holdout_test_mini.csv")
test = test.dropna()
test = pd.get_dummies(data[features], drop_first=True)

predictions = clf.predict(test)
my_predictions = pd.DataFrame(predictions, columns=['y'])
my_predictions.to_csv("team2-module2-predictions.csv", index=False)