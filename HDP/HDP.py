import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
import joblib

# Load the dataset
data = pd.read_csv("heart_cleveland_upload.csv")

# Separate features (X) and the target variable (y)
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# # Train a Decision Tree Classifier

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Train a k-Nearest Neighbors (KNN) Classifier

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Train a Logistic_Regression_model

lr_model = LogisticRegression() 
lr_model.fit(X_train, y_train)


# Save the trained models

joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(dt_model, 'decision_tree_model.pkl')
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(lr_model, 'logistic_regression.pkl')
