import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load and clean data
data = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv")
data = data.dropna()

# Select features and encode inputs
features = ["job", "default", "pdays", "poutcome"]
X = pd.get_dummies(data[features], drop_first=True)

# Encode target labels (yes/no to 1/0)
le = LabelEncoder()
y = le.fit_transform(data["y"])  # 'yes' -> 1, 'no' -> 0

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
print(clf.score(X_test, y_test))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()
plt.show()

# --- Mini holdout prediction ---
test = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank_holdout_test_mini.csv")
test = test.dropna()

# Match training features
test_X = pd.get_dummies(test[features], drop_first=True)
test_X = test_X.reindex(columns=X.columns, fill_value=0)

# Predict on holdout and save as 1/0
predictions = clf.predict(test_X)
my_predictions = pd.DataFrame(predictions, columns=['predictions'])
my_predictions.to_csv("team3-module2-predictions.csv", index=False)
